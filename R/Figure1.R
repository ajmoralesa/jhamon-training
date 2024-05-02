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

repes %>%
    group_by(tr_session) %>%
    summarise(totalrep = sum(reps)) %>%
    ggplot(aes(y = totalrep, x = tr_session)) +
    geom_bar(stat = "identity") +
    geom_label(aes(label = protocol$labs), hjust = 1.05, size = 3) +
    labs(x = "Training session", y = "Repetitions") +
    coord_flip() +
    theme_bw() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
    ) +
    ggsave("./reports/Figure_1.tiff", device = "tiff")
