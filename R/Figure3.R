# Create figure: two way SPM ANOVAS
library(reticulate)
library(stringr)
library(dplyr)
library(tibble)
library(forcats)
library(ggplot2)
library(tidyr)
library(arrow)

source("R/utils.R")

path_to_spmdict <- str_c(pthtres, "/training_spmdict")

spmdict <- py_load_object(path_to_spmdict)
f_time <- list()
f_group <- list()
f_interact <- list()

for (tr in seq_along(spmdict)) {
    vartime <- str_c("Ftime_", "tr", tr)
    vargroup <- str_c("Fgroup_", "tr", tr)
    varinteract <- str_c("Finter_", "tr", tr)
    f_time[[tr]] <- tibble(
        !!paste0(vartime, "_z") := spmdict[[tr]]$F_time$z,
        !!paste0(vartime, "_zs") := spmdict[[tr]]$F_time$zstar
    )

    f_group[[tr]] <- tibble(
        !!paste0(vargroup, "_z") := spmdict[[tr]]$F_group$z,
        !!paste0(vargroup, "_zs") := spmdict[[tr]]$F_time$zstar
    )

    f_interact[[tr]] <- tibble(
        !!paste0(varinteract, "_z") := spmdict[[tr]]$F_interaction$z,
        !!paste0(varinteract, "_zs") := spmdict[[tr]]$F_time$zstar
    )
}

f_stats <-
    bind_cols(f_time) %>%
    bind_cols(f_group) %>%
    bind_cols(f_interact)

# to long
f_long <- f_stats %>%
    rowid_to_column(var = "timepoint") %>%
    gather(key, value, -timepoint) %>%
    separate(col = key, into = c("aov", "tr", "var"), sep = "_") %>%
    mutate_if(purrr::is_character, as_factor)


zstars <- f_long %>%
    select(-timepoint) %>%
    filter(var == "zstar") %>%
    group_by(aov) %>%
    summarise(value = mean(value))

aovspmplt <- f_long %>%
    spread(var, value) %>%
    ggplot(aes(x = timepoint, y = z, group = tr)) +
    geom_line(aes(color = tr), size = 0.5) +
    geom_ribbon(aes(ymin = pmax(z, zs), ymax = zs, fill = tr), alpha = 0.1) +
    geom_hline(aes(yintercept = zs, color = tr), linetype = "dashed") +
    theme_bw() +
    scale_color_viridis_d(option = "plasma", direction = -1) +
    scale_fill_viridis_d(option = "plasma", direction = -1) +
    theme(
        legend.position = "right",
        axis.text = element_text(color = "black"),
        panel.grid = element_line(linetype = "longdash")
    ) +
    labs(y = "SPM{F}", x = "Knee angle (%)") +
    facet_grid(aov ~ ., scales = "free")


# Create figure grid with both SPM stats and curves
fig1 <- plot_grid(pltavg, aovspmplt, labels = c("A", "B"), rel_widths = c(2, 1), label_size = 12)
save_plot("./reports/figure1.jpeg", plot = fig1, ncol = 3, nrow = 2, base_asp = 1.1)


# # Elements for plots

# myplt <- list(
#     geom_line(aes(
#         x = timepoint, y = value, group = interaction(par, trses, rep, set),
#         colour = tr_group
#     ), alpha = 0.01),
#     geom_line(aes(
#         x = timepoint, y = avgses, group = interaction(par, trses, rep, set),
#         colour = tr_group
#     ), alpha = 0.01),
#     theme_bw(),
#     theme(
#         legend.position = c(0.1, 0.8),
#         legend.key.size = unit(0.3, "cm"),
#         legend.box.background = element_blank(),
#         panel.grid = element_line(linetype = "longdash")
#     )
# )



# myspmplt <- list(
#     geom_line(size = 0.5),
#     theme_bw(),
#     theme(
#         legend.position = "right",
#         axis.text = element_text(color = "black")
#     ),
#     theme(panel.grid = element_line(linetype = "longdash")),
#     labs(y = "SPM{t}", x = "Time (%)")
# )


# create_fig_torque_ttest <- function(dat, path_to_spmdict = "E:/_RESULTS/tor_spmdict.pkl") {
#     varplot <- dat %>%
#         filter(var %in% c("torque")) %>%
#         group_by(timepoint, trses, set, rep, var, tr_group) %>%
#         mutate(value = mean(value)) %>%
#         ungroup() %>%
#         group_by(timepoint, trses, var, tr_group) %>%
#         mutate(avgses = mean(value)) %>%
#         ungroup() %>%
#         ggplot() +
#         myplt +
#         labs(x = "Time (%)", y = "Torque (Nm)")




#     # SPM stats
#     spmdict <- py_load_object(filename = path_to_spmdict)
#     z <- spmdict$ti$z %>% tibble::enframe(name = NULL)
#     zs <- spmdict$ti$zstar


#     spmplot <- z %>%
#         ggplot(aes(x = seq(1, 101), y = value)) +
#         geom_ribbon(aes(ymin = pmax(z$value, zs), ymax = zs), alpha = 0.1) +
#         geom_hline(aes(yintercept = zs), linetype = "dashed") +
#         geom_ribbon(aes(ymin = pmin(z$value, zs * -1), ymax = zs * -1), alpha = 0.1) +
#         geom_hline(aes(yintercept = zs * -1), linetype = "dashed") +
#         myspmplt +
#         coord_cartesian(ylim = c(-40, 40))





#     fig1 <- plot_grid(varplot, spmplot, labels = c("A", "B"), rel_widths = c(1.3, 1), label_size = 12)

#     return(fig1)
# }



# create_fig_kneev_ttest <- function(dat, path_to_spmdict = "E:/_RESULTS/kneev_spmdict.pkl") {
#     varplot <- dat %>%
#         filter(var %in% c("knee_v")) %>%
#         filter(!(tr_group == "IK" & var == "knee_v" & value > 5)) %>%
#         filter(!(par == "04_guilhem" & trses %in% c("tr_6", "tr_4"))) %>%
#         mutate(value = case_when(
#             var %in% c("knee_v", "knee_ROM") & tr_group == "IK" ~ value * -1,
#             TRUE ~ value
#         )) %>%
#         group_by(timepoint, trses, set, rep, var, tr_group) %>%
#         mutate(value = mean(value)) %>%
#         ungroup() %>%
#         group_by(timepoint, trses, var, tr_group) %>%
#         mutate(avgses = mean(value)) %>%
#         ungroup() %>%
#         ggplot() +
#         myplt +
#         labs(x = "Time (%)", y = "Velocity (deg/s)")



#     # SPM stats

#     spmdict <- py_load_object(filename = path_to_spmdict)
#     z <- spmdict$ti$z %>% tibble::enframe(name = NULL)
#     zs <- spmdict$ti$zstar


#     spmplot <- z %>%
#         ggplot(aes(x = seq(1, 101), y = value)) +
#         myspmplt +
#         geom_ribbon(aes(ymin = pmax(z$value, zs), ymax = zs), alpha = 0.1) +
#         geom_hline(aes(yintercept = zs), linetype = "dashed") +
#         geom_ribbon(aes(ymin = pmin(z$value, zs * -1), ymax = zs * -1), alpha = 0.1) +
#         geom_hline(aes(yintercept = zs * -1), linetype = "dashed") +
#         coord_cartesian(ylim = c(-100, 100))


#     fig1 <- plot_grid(varplot, spmplot, labels = c("C", "D"), rel_widths = c(1.3, 1), label_size = 12)

#     return(fig1)
# }
