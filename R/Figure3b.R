library(ggplot2)
library(viridis)
library(arrow)
library(dplyr)

source("R/utils.R")

path_to_spmdict <- str_c(pthtres, "/trprogression_spmdict")

spmdict <- py_load_object(path_to_spmdict)


spmdict$torqueNH$tr_2$z

spm_nh <- list()
spm_ik <- list()

for (tr in seq_along(spmdict$torqueNH)) {
    spm_nh[[tr]] <- tibble(
        !!paste0("nh_", "z_", "tr", tr) := spmdict$torqueNH[[tr]]$z,
        !!paste0("nh_", "zs_", "tr", tr) := spmdict$torqueNH[[tr]]$zstar
    )

    spm_ik[[tr]] <- tibble(
        !!paste0("ik_", "z_", "tr", tr) := spmdict$torqueIK[[tr]]$z,
        !!paste0("ik_", "zs_", "tr", tr) := spmdict$torqueIK[[tr]]$zstar
    )
}

spm_all <- bind_cols(spm_nh, spm_ik)

# to long
spm_long <- spm_all %>%
    rowid_to_column(var = "timepoint") %>%
    pivot_longer(
        cols = -timepoint,
        names_to = c("tr_group", ".value", "tr"),
        names_sep = "_"
    ) |>
    mutate_if(purrr::is_character, as_factor)



plt_spm_long <- spm_long |>
    ggplot(aes(x = timepoint, y = z, group = tr)) +
    geom_line(aes(color = tr), size = 0.5) +
    geom_ribbon(aes(ymin = pmax(z, zs), ymax = zs, fill = tr), alpha = 0.1) +
    geom_hline(aes(yintercept = zs, color = tr), linetype = "dashed") +
    theme_bw() +
    theme(
        panel.grid = element_line(linetype = "longdash"),
        legend.position = "none",
        strip.background = element_blank(),
        strip.text.x = element_blank(),
        axis.text = element_text(color = "black")
    ) +
    labs(y = "SPM{F}", x = "Time (%)") +
    scale_color_viridis_d(option = "plasma", direction = -1) +
    scale_fill_viridis_d(option = "plasma", direction = -1) +
    facet_grid(. ~ tr_group) +
    theme_bw()

ggsave("./reports/figspm.png", plot = plt_spm_long, width = 10, height = 10)




trlong_torque() <- damecurvas(filename = "/training_results_torque", pathtofile = pthtres)

# Elements for both plots
myplt <- list()

pltavg <- trlong_torque %>%
    group_by(timepoint, trses, set, rep, var, tr_group) %>%
    mutate(avgrep = mean(value)) %>%
    ungroup() %>%
    group_by(timepoint, trses, var, tr_group) %>%
    mutate(avgses = mean(value)) %>%
    ungroup() %>%
    ggplot() +
    geom_line(aes(
        x = timepoint, y = avgrep, group = interaction(par, trses, rep, set),
        colour = as.numeric(tr_session)
    ), alpha = 0.01) +
    geom_line(aes(
        x = timepoint, y = avgses, group = interaction(par, trses, rep, set),
        colour = as.numeric(tr_session)
    )) +
    scale_color_viridis_c(option = "viridis", direction = 1) +
    labs(x = "Knee ROM (%)", y = "Torque (N.m)", color = "Session number") +
    facet_wrap(tr_group ~ .) +
    theme_bw() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "gray10", size = 0.25),
        legend.position = c(0.1, 0.8),
        legend.key.size = unit(0.3, "cm"),
        legend.box.background = element_blank(),
    )

# Save plot
timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
ggsave(paste0("./reports/figureTRAINING_", timestamp, ".png"), plot = pltavg)


# Plot of the two best repetitions for each session


bestreps <- trlong_torque |>
    group_by(par, trses, set) |>
    dplyr::filter(value == max(value)) |>
    ungroup() |>
    select(all_labels)

# All reps averaged
pltall <- trlong_torque |>
    dplyr::filter(par %in% c("jhamon01")) |>
    group_by(timepoint, trses, var, tr_group) %>%
    mutate(avgrep = mean(value)) |>
    ggplot() +
    geom_line(aes(
        x = timepoint,
        y = avgrep,
        colour = as.numeric(tr_session),
        group = interaction(par, trses, rep, set)
    ), alpha = 0.5) +
    scale_color_viridis_c(option = "viridis", direction = 1) +
    labs(title = "All reps, averaged by session")

pltdisc <- trlong_torque |>
    # dplyr::filter(par %in% c("jhamon01")) |>
    group_by(par, tr_session) %>%
    mutate(
        avgrep = mean(value),
    ) |>
    ungroup() |>
    mutate(
        avgses1 = mean(avgrep[tr_session == "tr_1"]),
        perchange = (avgrep * 100) / avgses1
    ) |>
    group_by(par, tr_session) |>
    summarise(
        perchange = mean(perchange)
    ) |>
    ggplot(aes(x = tr_session, y = perchange)) +
    geom_point()

timestamp2 <- format(Sys.time(), "%Y%m%d%H%M%S")

ggsave(paste0("./reports/figtorques", timestamp2, ".png"), plot = pltdisc)


perchange <- 100 * (avgrep / avgrep[tr_session == "tr_1"] - 1)

# Best reps per set averaged
trlong_torque() |>
    dplyr::filter(
        par %in% c("jhamon01"),
        all_labels %in% bestreps$all_labels
    ) |>
    group_by(timepoint, trses, var, tr_group) %>%
    mutate(avgrep = mean(value)) |>
    ggplot() +
    geom_line(aes(
        x = timepoint,
        y = avgrep,
        colour = as.numeric(tr_session),
        group = interaction(par, trses, rep, set)
    ), alpha = 0.5) +
    scale_color_viridis_c(option = "viridis", direction = 1) +
    labs(title = "BEST reps, averaged by session")
