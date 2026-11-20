
# for Figure 4
ql_us <- round( (sum(NAWS_N_data_laborz$QLAB17) - sum(NAWS_N_data_laborz$QLAB02))/sum(NAWS_N_data_laborz$QLAB02) *100, 1)
pl_us <- round( (sum(NAWS_N_data_laborz$WAGE17) - sum(NAWS_N_data_laborz$WAGE02 * 1.37))/sum(NAWS_N_data_laborz$WAGE02*1.37) *100, 1)

phil_us <- round( sum(PHIL_Na$phil/100 * NAWS_N_data_labora$QLAB02) / sum(NAWS_N_data_labora$QLAB02) *100, 1)
phil_lev_us <- phil_us * -1 / 100 * sum(NAWS_N_data_labora$QLAB02[NAWS_N_data_labora$PR9 %in% sel_pr]) / 1000 #('Mil Hours')

# ---- merge data with counties map to visualize ----
pr.poly <- st_read(file.path("in/pr.poly/pr_polygons.shp")) #pr_poly.R

# Census cartographic boundary files (state & county, 2018, 1:500k).
# Not included in the replication archive due to file size -- downloaded
# on first run and cached locally in in/ for subsequent runs.
get_cb_shapefile <- function(name, dest_dir = "in") {
  shp_dir  <- file.path(dest_dir, name)
  shp_path <- file.path(shp_dir, paste0(name, ".shp"))
  if (!file.exists(shp_path)) {
    dir.create(shp_dir, recursive = TRUE, showWarnings = FALSE)
    zip_path <- file.path(shp_dir, paste0(name, ".zip"))
    url <- paste0("https://www2.census.gov/geo/tiger/GENZ2018/shp/", name, ".zip")
    message("Downloading ", name, " from Census Bureau...")
    download.file(url, destfile = zip_path, mode = "wb", quiet = TRUE)
    unzip(zip_path, exdir = shp_dir)
    file.remove(zip_path)
  }
  st_read(shp_path)
}

#loaded in NLCD section
county_us  <- get_cb_shapefile("cb_2018_us_county_500k")
state_us   <- get_cb_shapefile("cb_2018_us_state_500k")
newcrs     <- st_crs("+proj=longlat +datum=WGS84")
county_us  <- st_transform(county_us, newcrs)
state_us   <- st_transform(state_us, newcrs)

ext              <- c(-124.5, -67, 20, 49.333) # CO-US
bbox <- st_bbox(c(xmin = ext[1], xmax = ext[2], ymin = ext[3], ymax = ext[4]), crs = st_crs(newcrs))
county <- county_us %>%
  filter(STATEFP!=15 & STATEFP!=02) %>%
  mutate(FIPS_ed = as.character(paste0(STATEFP, COUNTYFP))) %>%
  mutate(FIPS = as.numeric(GEOID)) %>%
  mutate(FIPS = case_when(FIPS==46102 ~ 46113, TRUE ~ FIPS)) %>%
  dplyr::select(STATEFP, FIPS, FIPS_ed, geometry) %>%
  st_crop(bbox) %>%
  mutate(STATEFP = as.numeric(STATEFP))

state_cous <- state_us %>%
  st_crop(bbox)

state_cous_ca <- state_cous %>%
  filter(STUSPS == "CA")

poly.cous      <- st_union(state_cous)
poly.cous.ca   <- st_union(state_cous_ca)

map    <-  county %>%
  full_join(FIPS2CZ) 

philz         <- PHIL_Nz
data_labor_cz <- NAWS_N_data_laborz

map.cz <- map %>%
  full_join(data_cz %>% dplyr::select(CZ, vcrop, p_agg, q_gr, q_fn, q_vg, q_ot, tgr, tfn, tvg, tot, s_immi, q, qc, HLAND2002_ltavg, HLAND2017_ltavg, phic)) %>%
  mutate(tfv = ifelse(tvg + tfn >15, tvg + tfn, 0)) %>%
  mutate(q_fv = ifelse(tfv!=0, (tvg/tfv * q_vg) + (tfn/tfv * q_fn),0),
         q_gr1 = ifelse(tgr >15, q_gr, NA),
         q_ot1 = ifelse(tot >15, q_ot, NA)) %>%
  full_join(philz %>% dplyr::select(CZ, phil)) %>%
  full_join(data_labor_cz %>% dplyr::select(CZ, QLAB02, VLAB2002_ltavg, QLAB17, VLAB2017_ltavg, ql, pl)) %>%
  mutate(yldz = q - qc) %>%
  mutate(phil_lev = phil/100 * QLAB02,
         plot.phil     = phil*-1,
         plot.phic     = phic*-1,
         plot.phic.lev = phic / (HLAND2002_ltavg/1000), # plotting as same as theory
         plot.phil.lev = phil_lev*-1/1000) %>% # '000 hours
  # input use intensities
  mutate(lab_int         = QLAB17 / HLAND2017_ltavg,
         p_lab_int       = ql - qc,
         p_lab_int_vcrop = vcrop - ql,
         p_lab_int_qcrop = q - ql,
         p_yield         = q - qc) %>%
  full_join(CZ2PR)

# ---- Plots for paper: March 2026 ----
# Figure 4 ----
# A ----
map.cz4 <- map.cz %>%
  filter(PR9 %in% sel_pr)

neg_breaks = 3
pos_breaks = 6
lower_val <- round(quantile(map.cz4$q, na.rm = TRUE, 0.05), 1)
upper_val <- round(quantile(map.cz4$q, na.rm = TRUE, 0.90), 1)

neg_int_breaks      <- seq(from = floor(lower_val), to = ceiling(0), length.out = neg_breaks)
pos_int_breaks      <- seq(from = 0, to = ceiling(upper_val), length.out = pos_breaks)
formatted_labels    <- c(round(neg_int_breaks,0), round(pos_int_breaks,0))
formatted_labels    <- unique(formatted_labels)
colors = c(brewer.pal(11, "RdBu")[3], brewer.pal(11, "RdBu")[3], brewer.pal(11, "RdBu")[3], brewer.pal(11, "RdBu")[4:11])

Q1 <- ggplot(data = map.cz4) +
  geom_sf(aes(fill = q), color = NA) +  
  geom_sf(data = pr.poly, fill = NA, color = "green", size=0.5) +
  geom_sf(data = poly.cous, fill = NA, color = "black") +
  scale_fill_gradientn(name = "",
                       colors = colors,
                       values = scales::rescale(c(lower_val, 0, upper_val)),
                       limits = c(lower_val, upper_val),
                       breaks  = formatted_labels,
                       labels  = formatted_labels, # Explicit labels
                       oob = scales::squish) +
  theme_void() +
  ggtitle("") +
  theme(plot.title    = element_text(hjust = 0.5, size = 12, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        legend.text   = element_text(size = 8), legend.position = "bottom") +
  guides(fill = guide_colorbar(barwidth = 12, # Adjust the width of the legend bar
                               barheight = 0.8))
Q1

# B ----
lower_val <- round(quantile(map.cz4$q_gr1, na.rm = TRUE, 0.05), 1)
upper_val <- round(quantile(map.cz4$q_gr1, na.rm = TRUE, 0.80), 1)

neg_int_breaks      <- seq(from = floor(lower_val), to = ceiling(0), length.out = neg_breaks)
pos_int_breaks      <- seq(from = 0, to = ceiling(upper_val), length.out = pos_breaks)
formatted_labels    <- c(round(neg_int_breaks,0), round(pos_int_breaks,0))
formatted_labels    <- unique(formatted_labels)
colors = c(brewer.pal(11, "RdBu")[3], brewer.pal(11, "RdBu")[3], brewer.pal(11, "RdBu")[3], brewer.pal(11, "RdBu")[4:11])

QGRz1 <- ggplot(data = map.cz4) +
  geom_sf(aes(fill = q_gr1), color = NA) +  
  geom_sf(data = pr.poly, fill = NA, color = "green", size=0.5) +
  geom_sf(data = poly.cous, fill = NA, color = "black") +
  scale_fill_gradientn(name = "",
                       colors = colors,
                       values = scales::rescale(c(lower_val, 0, upper_val)),
                       na.value = "white",
                       limits = c(lower_val, upper_val),
                       breaks  = formatted_labels,
                       labels  = formatted_labels, # Explicit labels
                       oob = scales::squish) +
  theme_void() +
  ggtitle(" ") +
  theme(plot.title    = element_text(hjust = 0.5, size = 12, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        legend.text   = element_text(size = 8), legend.position = "bottom") +
  guides(fill = guide_colorbar(barwidth = 12, # Adjust the width of the legend bar
                               barheight = 0.8))

# C ----
neg_breaks = 4
pos_breaks = 4
map.cz4$q_fv1 = ifelse(map.cz4$q_fv==0, NA, map.cz4$q_fv)

lower_val <- round(quantile(map.cz4$q_fv1, na.rm = TRUE, 0), 1)
upper_val <- round(quantile(map.cz4$q_fv1, na.rm = TRUE, 0.95), 1)

neg_int_breaks      <- seq(from = floor(lower_val), to = ceiling(0), length.out = neg_breaks)
pos_int_breaks      <- seq(from = 0, to = ceiling(upper_val), length.out = pos_breaks)
formatted_labels    <- c(round(neg_int_breaks,0), round(pos_int_breaks,0))
formatted_labels    <- unique(formatted_labels)
colors = c(rep(brewer.pal(11, "RdBu")[3],2), brewer.pal(11, "RdBu")[4:6], brewer.pal(11, "RdBu")[10], brewer.pal(11, "RdBu")[10:11])

Q3 <- ggplot(data = map.cz4) +
  geom_sf(aes(fill = q_fv1), color = NA) +  
  geom_sf(data = pr.poly, fill = NA, color = "green", size=0.5) +
  geom_sf(data = poly.cous, fill = NA, color = "black") +
  scale_fill_gradientn(name = "",
                       colors = brewer.pal(11, "RdBu"),
                       values = scales::rescale(c(lower_val, 0, upper_val)),
                       na.value = "white",
                       limits = c(lower_val, upper_val),
                       breaks  = formatted_labels,
                       labels  = formatted_labels, # Explicit labels
                       oob = scales::squish) +
  theme_void() +
  ggtitle("") +
  theme(plot.title    = element_text(hjust = 0.5, size = 12, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        legend.text   = element_text(size = 8), legend.position = "bottom") +
  guides(fill = guide_colorbar(barwidth = 12, # Adjust the width of the legend bar
                               barheight = 0.8))
Q3

# D ----
neg_breaks = 4
pos_breaks = 5

lower_val <- round(quantile(map.cz4$q_ot1, na.rm = TRUE, 0), 1)
upper_val <- round(quantile(map.cz4$q_ot1, na.rm = TRUE, 0.95), 1)

neg_int_breaks      <- seq(from = floor(lower_val), to = ceiling(0), length.out = neg_breaks)
pos_int_breaks      <- seq(from = 0, to = ceiling(upper_val), length.out = pos_breaks)
formatted_labels    <- c(round(neg_int_breaks,0), round(pos_int_breaks,0))
formatted_labels    <- unique(formatted_labels)
colors = c(rep(brewer.pal(11, "RdBu")[3],2), brewer.pal(11, "RdBu")[4:6], brewer.pal(11, "RdBu")[10], brewer.pal(11, "RdBu")[10:11])

Q4 <- ggplot(data = map.cz4) +
  geom_sf(aes(fill = q_ot1), color = NA) +  
  geom_sf(data = pr.poly, fill = NA, color = "green", size=0.5) +
  geom_sf(data = poly.cous, fill = NA, color = "black") +
  scale_fill_gradientn(name = "",
                       colors = brewer.pal(11, "RdBu"),
                       values = scales::rescale(c(lower_val, 0, upper_val)),
                       na.value = "white",
                       limits = c(lower_val, upper_val),
                       breaks  = formatted_labels,
                       labels  = formatted_labels, # Explicit labels
                       oob = scales::squish) +
  theme_void() +
  ggtitle("") +
  theme(plot.title    = element_text(hjust = 0.5, size = 12, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        legend.text   = element_text(size = 8), legend.position = "bottom") +
  guides(fill = guide_colorbar(barwidth = 12, # Adjust the width of the legend bar
                               barheight = 0.8))
Q4

# combine ----
png(paste0("plots/Figure4.png"), units = "in", width=8, height=6, res=300) 
print(
  ggdraw() + 
  draw_plot(Q1,    x=0.01,  y = 0.5, width = 0.495, height = 0.48) +
  draw_label("A: % Change in all crop production in CZs (2002 - 2017)", 
             x = 0.25, y = 0.97, fontface = "bold", size = 10, hjust = 0.5) +
  draw_label("(National change: 26.9%)", 
             x = 0.25, y = 0.95, size = 8, hjust = 0.5) +
  draw_plot(QGRz1,    x=0.5,   y = 0.5, width = 0.495, height = 0.48) +
  draw_label("B: % Change in grain production in CZs (2002 - 2017)", 
             x = 0.75, y = 0.97, fontface = "bold", size = 10, hjust = 0.5) +
  draw_label("(National change: 51.1%)", 
             x = 0.75, y = 0.95, size = 8, hjust = 0.5) +
  draw_plot(Q3,    x=0.01,  y = 0.01, width = 0.495, height = 0.48) +
  draw_label("C: % Change in fruits, nuts & veg. production in CZs (2002 - 2017)", 
             x = 0.25, y = 0.49, fontface = "bold", size = 9, hjust = 0.5) +
  draw_label("(National change: -4%)", 
             x = 0.25, y = 0.46, size = 8, hjust = 0.5) +
  draw_plot(Q4,    x=0.5,  y = 0.01, width = 0.495, height = 0.48) +
  draw_label("D: % Change in other crop production in CZs (2002 - 2017)",  
             x = 0.75, y = 0.49, fontface = "bold", size = 10, hjust = 0.5) +
  draw_label("(National change: 14.9%)", 
             x = 0.75, y = 0.46, size = 8, hjust = 0.5) 
)
dev.off()

# Figure 5 ----
# A ----
neg_breaks = 3
pos_breaks = 6
lower_val <- round(quantile(map.cz4$ql, na.rm = TRUE, 0.05), 1)
upper_val <- round(quantile(map.cz4$ql, na.rm = TRUE, 0.90), 1)

neg_int_breaks      <- seq(from = floor(lower_val), to = ceiling(0), length.out = neg_breaks)
pos_int_breaks      <- seq(from = 0, to = ceiling(upper_val), length.out = pos_breaks)
formatted_labels    <- c(round(neg_int_breaks,0), round(pos_int_breaks,0))
formatted_labels    <- unique(formatted_labels)
colors = c(brewer.pal(11, "RdBu")[3], brewer.pal(11, "RdBu")[3], brewer.pal(11, "RdBu")[3], brewer.pal(11, "RdBu")[4:11])

Q1 <- ggplot(data = map.cz4) +
  geom_sf(aes(fill = ql), color = NA) +  
  geom_sf(data = pr.poly, fill = NA, color = "green", size=0.5) +
  geom_sf(data = poly.cous, fill = NA, color = "black") +
  scale_fill_gradientn(name = "",
                       colors = colors,
                       values = scales::rescale(c(lower_val, 0, upper_val)),
                       limits = c(lower_val, upper_val),
                       breaks  = formatted_labels,
                       labels  = formatted_labels, # Explicit labels
                       oob = scales::squish) +
  theme_void() +
  ggtitle("") +
  theme(plot.title    = element_text(hjust = 0.5, size = 12, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        legend.text   = element_text(size = 8), legend.position = "bottom") +
  guides(fill = guide_colorbar(barwidth = 12, # Adjust the width of the legend bar
                               barheight = 0.8))
Q1
# B ----
lower_val <- round(quantile(map.cz4$pl, na.rm = TRUE, 0), 1)
upper_val <- round(quantile(map.cz4$pl, na.rm = TRUE, 0.975), 1)
upper_val <- 70
# neg_int_breaks      <- seq(from = floor(lower_val), to = ceiling(0), length.out = neg_breaks)
# pos_int_breaks      <- seq(from = 0, to = ceiling(upper_val), length.out = pos_breaks)
# formatted_labels    <- c(round(neg_int_breaks,0), round(pos_int_breaks,0))
# formatted_labels    <- unique(formatted_labels)
colors = brewer.pal(9, "Blues")

Q2 <- ggplot(data = map.cz4) +
  geom_sf(aes(fill = pl), color = NA) +  
  geom_sf(data = pr.poly, fill = NA, color = "green", size=0.5) +
  geom_sf(data = poly.cous, fill = NA, color = "black") +
  scale_fill_gradientn(name = "",
                       colors = colors,
                       limits = c(lower_val, upper_val),
                       # breaks  = formatted_labels,
                       # labels  = formatted_labels, # Explicit labels
                       oob = scales::squish) +
  theme_void() +
  ggtitle("") +
  theme(plot.title    = element_text(hjust = 0.5, size = 12, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        legend.text   = element_text(size = 8), legend.position = "bottom") +
  guides(fill = guide_colorbar(barwidth = 12, # Adjust the width of the legend bar
                               barheight = 0.8))
Q2

# C ----
upper_val <- 0
lower_val <- round(quantile(map.cz4$phil*-1, na.rm = TRUE, 0.025), 1)
colors = brewer.pal(9, "Reds")

# lower_val <- round(quantile(map.cz4$phil*-1, na.rm = TRUE, 0.02), 1)
# upper_val <- round(quantile(map.cz4$phil*-1, na.rm = TRUE, 0.95), 1)
# 
# neg_int_breaks      <- seq(from = floor(lower_val), to = ceiling(0), length.out = neg_breaks)
# pos_int_breaks      <- seq(from = 0, to = ceiling(upper_val), length.out = pos_breaks)
# formatted_labels    <- c(round(neg_int_breaks,0), round(pos_int_breaks,0))
# formatted_labels    <- unique(formatted_labels)
# colors = c(brewer.pal(11, "RdBu")[1:6], rep(brewer.pal(11, "RdBu")[7],5))

PHIL1 <- ggplot(data = map.cz4) +
  geom_sf(aes(fill = phil*-1), color = NA) +  
  geom_sf(data = pr.poly, fill = NA, color = "green", size=0.5) +
  geom_sf(data = poly.cous, fill = NA, color = "black") +
  scale_fill_gradientn(name = "",
                       colors = rev(colors),
                       limits = c(lower_val, upper_val),
                       # breaks  = formatted_labels,
                       # labels  = formatted_labels, # Explicit labels
                       oob = scales::squish) +
  theme_void() +
  ggtitle("") +
  theme(plot.title    = element_text(hjust = 0.5, size = 12, face="bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        legend.text   = element_text(size = 8), legend.position = "bottom") +
  guides(fill = guide_colorbar(barwidth = 12, # Adjust the width of the legend bar
                               barheight = 0.8))

# D ----
neg_breaks = 6
pos_breaks = 2

lower_val <- round(quantile(map.cz4$plot.phil.lev, na.rm = TRUE, 0.1), 1)
upper_val <- round(quantile(map.cz4$plot.phil.lev, na.rm = TRUE, 0.98), 1)
neg_int_breaks      <- seq(from = floor(lower_val), to = ceiling(0), length.out = neg_breaks)
pos_int_breaks      <- seq(from = 0, to = ceiling(upper_val), length.out = pos_breaks)
formatted_labels    <- c(round(neg_int_breaks,0), round(pos_int_breaks,0))
formatted_labels    <- unique(formatted_labels)
colors = c(brewer.pal(11, "RdBu")[1:6], rep(brewer.pal(11, "RdBu")[7],5))

PHIL.lev1 <- ggplot(data = map.cz4) +
  geom_sf(aes(fill = plot.phil.lev), color = NA) +  
  geom_sf(data = pr.poly, fill = NA, color = "green", size=0.5) +
  geom_sf(data = poly.cous, fill = NA, color = "black") +
  scale_fill_gradientn(name = "",
                       colors = colors,
                       values = scales::rescale(c(lower_val, 0, upper_val)),
                       limits = c(lower_val, upper_val),
                       breaks  = formatted_labels,
                       labels  = formatted_labels, # Explicit labels
                       oob = scales::squish) +
  theme_void() +
  ggtitle("") +
  theme(plot.title    = element_text(hjust = 0.5, size = 12, face="bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        legend.text   = element_text(size = 8), legend.position = "bottom") +
  guides(fill = guide_colorbar(barwidth = 12, # Adjust the width of the legend bar
                               barheight = 0.8))

# combine ----
png(paste0("plots/Figure5.png"), units = "in", width=8, height=6, res=300) 
print(
  ggdraw() + 
  draw_plot(Q1,    x=0.01,  y = 0.5, width = 0.495, height = 0.48) +
  draw_label("A: % Change in Employment (labor hours) (2002 - 2017)", 
             x = 0.25, y = 0.98, fontface = "bold", size = 10, hjust = 0.5) +
  draw_label("(National change: 8.6%)", x = 0.25, y = 0.95, size = 8, hjust = 0.5) +
  
  draw_plot(Q2,    x=0.5,   y = 0.5, width = 0.495, height = 0.48) +
  draw_label("B: % Change in Real Wages ($/hr) (2002 - 2017)", 
             x = 0.75, y = 0.98, fontface = "bold", size = 10, hjust = 0.5) +
  draw_label("(National change: 20.8%)", x = 0.75, y = 0.95, size = 8, hjust = 0.5) +
  
  draw_plot(PHIL1,    x=0.01,  y = 0.01, width = 0.495, height = 0.48) +
  draw_label("C: % Change in Supply of Labor Hours (2002 - 2017)", 
             x = 0.25, y = 0.48, fontface = "bold", size = 10, hjust = 0.5) +
  draw_label("(National change: -12.6%)", x = 0.25, y = 0.45, size = 8, hjust = 0.5) +
  
  draw_plot(PHIL.lev1,    x=0.5,  y = 0.01, width = 0.495, height = 0.48) +
  draw_label("D: Change in Labor Hours ('000 Hrs) (2002 - 2017)",  
             x = 0.75, y = 0.48, fontface = "bold", size = 10, hjust = 0.5) +
  draw_label("(National change: -180,200,000 Hours)", x = 0.75, y = 0.45, size = 8, hjust = 0.5)
)
dev.off()




