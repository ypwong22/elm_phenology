

########################################
# Filter EN (evergreen needleleaf sites)
########################################
get_EN_sites <- function(){
    EN_sites = list.files("/lustre/haven/proj/UTK0134/DATA/Vegetation/PhenoCam_V2_1674/data/data_record_4", "*_EN_*")
    EN_sites = unique(unlist(lapply(strsplit(EN_sites, "_"), "[[", 1)))
    return (EN_sites)
}

