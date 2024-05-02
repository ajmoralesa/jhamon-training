library(stringr)
library(feather)
library(arrow)
library(forcats)
library(dplyr)
library(ggplot2)
library(viridis)
library(reticulate)
library(cowplot)

source("R/utils.R")


# Load data
trlong_torque <- damecurvas(filename = "training_results_torque", pathtofile = pthtres)
trlong_velocity <- damecurvas(filename = "training_results_velocity", pathtofile = pthtres)

# Elements for both plots
myplt <- list(
    geom_line(aes(
        x = timepoint, y = value, group = interaction(par, trses, rep, set),
        colour = tr_group
    ), alpha = 0.01),
    theme_classic(),
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "gray10", size = 0.25),
        legend.position = c(0.1, 0.8),
        legend.key.size = unit(1, "cm"),
        legend.box.background = element_blank(),
    )
)

# Torque plot
torque_plot <- trlong_torque %>%
    group_by(timepoint, trses, set, rep, var, tr_group) %>%
    mutate(value = mean(value)) %>%
    ungroup() %>%
    ggplot() +
    scale_color_viridis(discrete = TRUE, option = "cividis") +
    myplt +
    guides(
        colour = guide_legend(
            override.aes = list(alpha = 1, size = 1.5)
        )
    ) +
    labs(x = "Time (%)", y = "Torque (Nm)", colour = "Training Group")

# SPM results
path_to_tor_spmdict <- str_c(pthtres, "tor_spmdict.pkl")
tor_spmdict <- py_load_object(path_to_tor_spmdict)

z <- tor_spmdict$ti$z %>% tibble::enframe(name = NULL)
zs <- tor_spmdict$ti$zstar

myspmplt <- list(
    geom_line(size = 0.5),
    theme_classic(),
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "gray10", size = 0.25),
    ),
    theme(
        legend.position = "right",
        axis.text = element_text(color = "black")
    ),
    labs(y = "SPM{t}", x = "Time (%)")
)

# Plot the z-scores and the data
# First, set the z-scores
zs <- 2
# Next, create the plot
tor_spmplot <- z %>%
    ggplot(aes(x = seq(1, 101), y = value)) +
    # Plot the z-scores
    geom_ribbon(aes(ymin = pmax(z$value, zs), ymax = zs), alpha = 0.1) +
    geom_hline(aes(yintercept = zs), linetype = "dashed") +
    geom_ribbon(aes(ymin = pmin(z$value, zs * -1), ymax = zs * -1), alpha = 0.1) +
    geom_hline(aes(yintercept = zs * -1), linetype = "dashed") +
    # Plot the data
    myspmplt +
    coord_cartesian(ylim = c(-40, 40))

fig2_upper <- plot_grid(torque_plot, tor_spmplot, labels = c("A", "B"), rel_widths = c(1.3, 1), label_size = 12)



# Velocity plot
velocity_plot <- trlong_velocity |>
    # dplyr::filter(
    #     !(`par` == "jhamon04" & trses == "tr_6") |
    #         !(`par` == "jhamon04" & trses == "tr_6" & set == "set_3" & rep == "rep_3") |
    #         !(`par` == "jhamon04" & trses == "tr_6" & set == "set_3" & rep == "rep_1") |
    #         !(`par` == "jhamon04" & trses == "tr_6" & set == "set_1" & rep == "rep_4") |
    #         !(`par` == "jhamon04" & trses == "tr_6" & set == "set_4" & rep == "rep_6") |
    #         !(`par` == "jhamon04" & trses == "tr_6" & set == "set_4" & rep == "rep_4") |
    #         !(`par` == "jhamon04" & trses == "tr_6" & set == "set_4" & rep == "rep_5") |
    #         !(`par` == "jhamon04" & trses == "tr_6" & set == "set_4" & rep == "rep_2") |
    #         !(`par` == "jhamon04" & trses == "tr_4" & set == "set_2" & rep == "rep_3") |
    #         !(`par` == "jhamon04" & trses == "tr_4" & set == "set_2" & rep == "rep_4") |
    #         !(`par` == "jhamon04" & trses == "tr_4" & set == "set_1" & rep == "rep_4") |
    #         !(`par` == "jhamon03" & trses == "tr_15" & set == "set_5" & rep == "rep_3") |
    #         !(`par` == "jhamon03" & trses == "tr_15" & set == "set_2" & rep == "rep_4") |
    #         !(`par` == "jhamon03" & trses == "tr_15" & set == "set_3" & rep == "rep_3") |
    #         !(`par` == "jhamon03" & trses == "tr_15" & set == "set_3" & rep == "rep_7") |
    #         !(`par` == "jhamon03" & trses == "tr_15" & set == "set_4" & rep == "rep_6") |
    #         !(`par` == "jhamon02" & trses == "tr_6" & set == "set_1" & rep == "rep_3") |
    #         !(`par` == "jhamon02" & trses == "tr_6" & set == "set_4" & rep == "rep_3") |
    #         !(`par` == "jhamon02" & trses == "tr_6" & set == "set_4" & rep == "rep_4") |
    #         !(`par` == "jhamon02" & trses == "tr_6" & set == "set_2" & rep == "rep_2") |
    #         !(`par` == "jhamon06" & trses == "tr_6" & set == "set_1" & rep == "rep_6")
    # ) |>
    # dplyr::filter((tr_group == "NH" & value < -10)) |>
    group_by(timepoint, trses, set, rep, var, tr_group) %>%
    mutate(value = mean(value)) %>%
    ungroup() %>%
    ggplot() +
    scale_color_viridis(discrete = TRUE, option = "cividis") +
    myplt +
    guides(
        colour = guide_legend(
            override.aes = list(alpha = 1, size = 1.5)
        )
    ) +
    coord_cartesian(ylim = c(-10, 200)) +
    labs(x = "Time (%)", y = "Velocity (deg/s)", colour = "Training Group")

# dplyr::filter(!(tr_group == "IK" & var == "knee_v" & value > 5)) |>
#     mutate(value = case_when(
#         var %in% c("knee_v", "knee_ROM") & tr_group == "IK" ~ value * -1,
#         TRUE ~ value
#     )) |>
#     group_by(timepoint, trses, set, rep, var, tr_group) %>%
#     mutate(value = mean(value)) %>%
#     ungroup() %>%
#     ggplot() +
#     myplt +
#     scale_color_viridis(discrete = TRUE, option = "cividis") +
#     guides(
#         colour = guide_legend(
#             override.aes = list(alpha = 1, size = 1.5)
#         )
#     ) +
#     labs(x = "Time (%)", y = "Velocity (deg/s)", colour = "Training Group")


# SPM stats
kneev_spmdict <- py_load_object(filename = str_c(pthtres, "kneev_spmdict.pkl"))
z <- kneev_spmdict$ti$z %>% tibble::enframe(name = NULL)
zs <- kneev_spmdict$ti$zstar

kneev_spmplot <- z %>%
    ggplot(aes(x = seq(1, 101), y = value)) +
    myspmplt +
    geom_ribbon(aes(ymin = pmax(z$value, zs), ymax = zs), alpha = 0.1) +
    geom_hline(aes(yintercept = zs), linetype = "dashed") +
    geom_ribbon(aes(ymin = pmin(z$value, zs * -1), ymax = zs * -1), alpha = 0.1) +
    geom_hline(aes(yintercept = zs * -1), linetype = "dashed") +
    coord_cartesian(ylim = c(-100, 100))

fig2_bottom <- plot_grid(velocity_plot, kneev_spmplot, labels = c("C", "D"), rel_widths = c(1.3, 1), label_size = 12)


# Figure 2
fig2 <- plot_grid(fig2_upper, fig2_bottom, ncol = 1, nrow = 2)

# Generate a timestamp
timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")

# Save the plot
save_plot(paste0("./reports/Figure_2_", timestamp, ".png"), fig2, ncol = 2, nrow = 2)
