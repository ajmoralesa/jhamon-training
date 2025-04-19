# Create df with the number of reps per tr_session set
library(ggplot2)
library(magrittr)
library(dplyr)
library(stringr)
library(forcats)


repes <- tibble(
    tr_session = c(
        rep(str_c("tr_", 1), 3),
        rep(str_c("tr_", 2:5), each = 4),
        rep(str_c("tr_", 6:11), each = 5),
        rep(str_c("tr_", 12:15), each = 6)
    ),
    set = c(
        str_c("set_", 1:3),
        rep(str_c("set_", 1:4), times = 4),
        rep(str_c("set_", 1:5), times = 6),
        rep(str_c("set_", 1:6), times = 4)
    ),
    reps = c(rep(5, 19), rep(6, 15), rep(8, 39))
)
repes$tr_session <- as_factor(repes$tr_session)

protocol <- tibble(
    tr_session = str_c("tr_", 1:15),
    labs = c(
        str_c("3 sets x 5 reps"),
        rep(str_c("4 sets x 5 reps"), 4),
        rep(str_c("5 sets x 6 reps"), 3),
        rep(str_c("5 sets x 8 reps"), 3),
        rep(str_c("6 sets x 8 reps"), 4)
    )
)

# Define weekly boundaries for background rectangles
week_mapping <- tibble(
    tr_session = str_c("tr_", 1:15),
    week = c(rep(1:7, each = 2), 8) # Assign 2 sessions/week, last week gets 1
)
# Ensure factor levels match the main data for correct numeric mapping
week_mapping$tr_session <- factor(week_mapping$tr_session, levels = levels(repes$tr_session))

# Calculate numeric boundaries based on factor levels
rect_data <- week_mapping %>%
    mutate(session_num = as.numeric(tr_session)) %>%
    group_by(week) %>%
    summarise(
        xmin = min(session_num) - 0.5, # Offset by 0.5 for full bar coverage
        xmax = max(session_num) + 0.5,
        .groups = "drop"
    ) %>%
    mutate(
        ymin = -Inf, # Span the entire height
        ymax = Inf,
        # Alternate fill color based on even/odd week number
        fill_color = factor(week %% 2)
    )

# Create the plot
plot_data <- repes %>%
    group_by(tr_session) %>%
    summarise(totalrep = sum(reps))

ggplot(plot_data, aes(y = totalrep, x = tr_session)) +
    # Add background rectangles BEFORE bars
    geom_rect(
        data = rect_data,
        aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill_color),
        inherit.aes = FALSE, # Don't inherit main aes mappings
        alpha = 0.2 # Set transparency
    ) +
    scale_fill_manual(values = c("0" = "grey80", "1" = "grey90"), guide = "none") + # Define colors and hide legend
    geom_bar(stat = "identity") +
    geom_label(aes(label = protocol$labs), hjust = 1.05, size = 3) +
    labs(x = "Training session", y = "Repetitions") +
    coord_flip() +
    theme_bw() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
    )

ggsave("./reports/Figure_1.png", device = "png")
