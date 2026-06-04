library(stringr)
library(arrow)
library(forcats)

# Digested results live in the paper's NAS project folder (single source of
# truth; mirrors RESULTS_TRAINING_PATH in jhamon_training/pathutils.py).
# Older external-drive locations (kept for reference): "E:/_RESULTS_TRAINING/",
# "/Volumes/jHamON/_RESULTS_TRAINING".
pthtres <- "/Users/amorales/SynologyDrive/perso/RECHERCHE/Projects/2019_jHamON/_RESULTS_TRAINING"

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
