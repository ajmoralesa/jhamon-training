library(stringr)
library(arrow)
library(forcats)

# pthtres <- "E:/_RESULTS_TRAINING/"
pthtres <- "/Volumes/jHamON/_RESULTS_TRAINING"

damecurvas <- function(filename, pathtofile) {
    path <- str_c(pathtofile, filename)
    df <- tibble::as_tibble(read_feather(path))

    df$trses <- as_factor(df$trses)

    # Reorder tr_session labels factors to avoid 10 first
    tr_num <- seq_along(levels(df$trses))
    new_tr <- vector()
    for (ii in tr_num) {
        new_tr[ii] <- str_c("tr_", ii)
    }
    df$tr_session <- fct_relevel(df$trses, new_tr)

    return(df)
}
