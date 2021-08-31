# EDA

pacman::p_load(tidyverse, psych, gridExtra, tseries, broom, PCAmixdata)
setwd('~/Documents/GitHub/DSC-424---Multivariate/Final Project')
df <- as_tibble(read.csv('BankChurners.csv', header = T))
df <- df %>%
  select(-Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1, -Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2)
dim(df)
sum(is.na(df))
table(df$Attrition_Flag)
glimpse(df)
summary(df)

p1 <- df %>%
  ggplot(aes(x = Gender, fill = Attrition_Flag)) +
  geom_bar(position = "fill") +
  stat_count(geom = "text", 
             aes(label = stat(count)),
             position = position_fill(vjust=0.5), colour="white") +
  theme_minimal() +
  theme(legend.position = "none")


p2 <- df %>%
  ggplot(aes(x = Education_Level, fill = Attrition_Flag)) +
  geom_bar(position = "fill") +
  stat_count(geom = "text", 
             aes(label = stat(count)),
             position = position_fill(vjust=0.5), colour="white") +
  scale_x_discrete(guide = guide_axis(n.dodge = 2)) +
  theme_minimal() +
  theme(legend.position = "none")

p3 <- df %>%
  ggplot(aes(x = Marital_Status, fill = Attrition_Flag)) +
  geom_bar(position = "fill") +
  stat_count(geom = "text", 
             aes(label = stat(count)),
             position = position_fill(vjust=0.5), colour="white") +
  theme_minimal() +
  theme(legend.position = "none")

p4 <- df %>%
  ggplot(aes(x = Income_Category, fill = Attrition_Flag)) +
  geom_bar(position = "fill") +
  stat_count(geom = "text", 
             aes(label = stat(count)),
             position = position_fill(vjust=0.5), colour="white") +
  scale_x_discrete(guide = guide_axis(n.dodge = 2)) +
  theme_minimal() +
  theme(legend.position = "none")

p5 <- df %>%
  ggplot(aes(x = Card_Category, fill = Attrition_Flag)) +
  geom_bar(position = "fill") +
  stat_count(geom = "text", 
             aes(label = stat(count)),
             position = position_fill(vjust=0.5), colour="white") +
  theme_minimal()

grid.arrange(p1, p2, p3, p4, p5, nrow = 2)

dat <- data.frame(Variable = character(),ChiSq = numeric(), DF = numeric(), PVal = numeric())
for (name in names(df %>% select(where(is.character), - Attrition_Flag))){
  chiTest <- chisq.test(df$Attrition_Flag, df[[name]])
  dat <- rbind(dat, data.frame(Variable = name, ChiSq = chiTest$statistic, DF = chiTest$parameter, PVal = chiTest$p.value))
}
DT::datatable(dat, rownames = FALSE)

df %>%
  select(-where(is.character), -CLIENTNUM) %>%
  describe() %>%
  DT::datatable(options = list(scrollX = TRUE))

outlier.detection <- function(vector){
  ind <- boxplot.stats(vector)$out
  if (length(ind) > 0){
    paste0('There are ', length(ind), ' outliers.')
  }
}
for (con in names(df %>% select(-where(is.character), -CLIENTNUM))) {
  vec <- as.matrix(df %>% select(.data[[con]]))
  print(paste0(con, ':', outlier.detection(vec)))
}

dat <- data.frame(Variable = character(),ChiSq = numeric(), DF = numeric(), PVal = numeric())
for (name in names(df %>% select(-where(is.character), - CLIENTNUM))){
  jbTest <- jarque.bera.test(df[[name]])
  dat <- rbind(dat, data.frame(Variable = name, ChiSq = round(jbTest$statistic,4), DF = jbTest$parameter, PVal = round(jbTest$p.value, 4)))
}
DT::datatable(dat, rownames = FALSE,options = list(scrollX = TRUE))

p1 <- df %>%
  ggplot() +
  geom_density(aes(x = Customer_Age, col = Attrition_Flag)) +
  theme_minimal() +
  theme(legend.position = "none")

p2 <- df %>%
  ggplot() +
  geom_density(aes(x = Dependent_count, col = Attrition_Flag)) +
  theme_minimal() + theme(legend.position = "none")

p3 <- df %>%
  ggplot() +
  geom_density(aes(x = Months_on_book, col = Attrition_Flag)) +
  theme_minimal()+ theme(legend.position = "none")

p4 <- df %>%
  ggplot() +
  geom_density(aes(x = Total_Relationship_Count, col = Attrition_Flag)) +
  theme_minimal()+ theme(legend.position = "none")

p5 <- df %>%
  ggplot() +
  geom_density(aes(x = Months_Inactive_12_mon, col = Attrition_Flag)) +
  theme_minimal()+ theme(legend.position = "none")

p6 <- df %>%
  ggplot() +
  geom_density(aes(x = Contacts_Count_12_mon, col = Attrition_Flag)) +
  theme_minimal()+ theme(legend.position = "none")

p7 <- df %>%
  ggplot() +
  geom_density(aes(x = Credit_Limit, col = Attrition_Flag)) +
  theme_minimal()+ theme(legend.position = "none")

p8 <- df %>%
  ggplot() +
  geom_density(aes(x = Total_Revolving_Bal, col = Attrition_Flag)) +
  theme_minimal()+ theme(legend.position = "none")

p9 <- df %>%
  ggplot() +
  geom_density(aes(x = Avg_Open_To_Buy, col = Attrition_Flag)) +
  theme_minimal()+ theme(legend.position = "none")

p10 <- df %>%
  ggplot() +
  geom_density(aes(x = Total_Amt_Chng_Q4_Q1, col = Attrition_Flag)) +
  theme_minimal()+ theme(legend.position = "none")

p11 <- df %>%
  ggplot() +
  geom_density(aes(x = Total_Trans_Amt, col = Attrition_Flag)) +
  theme_minimal()+ theme(legend.position = "none")

p12 <- df %>%
  ggplot() +
  geom_density(aes(x = Total_Trans_Ct, col = Attrition_Flag)) +
  theme_minimal()+ theme(legend.position = "none")

p13 <- df %>%
  ggplot() +
  geom_density(aes(x = Total_Ct_Chng_Q4_Q1, col = Attrition_Flag)) +
  theme_minimal()+ theme(legend.position = "none")

p14 <- df %>%
  ggplot() +
  geom_density(aes(x = Avg_Utilization_Ratio, col = Attrition_Flag)) +
  theme_minimal()

grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, nrow = 5)

df %>%
  select(-where(is.character), -CLIENTNUM) %>%
  pairs.panels()

dat <- data.frame(Variable = character(),F = numeric(), PVal = numeric())
for (name in names(df %>% select(-where(is.character), -CLIENTNUM))){
  aovTest <- tidy(aov(df[[name]] ~ df$Attrition_Flag))
  dat <- rbind(dat, data.frame(Variable = name, F = round(aovTest$statistic,4), PVal = round(aovTest$p.value, 4)))
}
dat <- na.omit(dat)
DT::datatable(dat, rownames = FALSE)

df1 <- df %>%
  mutate_if(is.character, as.factor)
df.quanti <- df1 %>% select(-where(is.factor), -CLIENTNUM) %>% mutate_all(as.numeric) %>% as.data.frame()
df.quali <- df1 %>% select(where(is.factor),-Attrition_Flag) %>% as.data.frame()
pca <- PCAmix(X.quanti = scale(df.quanti), X.quali = df.quali, rename.level = TRUE,graph=FALSE)
DT::datatable(pca$eig)
pca$eig %>%
  as_tibble() %>%
  mutate(PC = 1:31) %>%
  ggplot(aes(x = as.factor(PC), y = Proportion, group = 1)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  xlab('PC')

par(mfrow=c(2,2))
plot(pca, choice = "ind", coloring.ind = df.quali$Income_Category,label = FALSE, posleg="bottomright", main="(a) Observations")
plot(pca,choice="levels", main="(b) Levels")
plot(pca,choice="cor",main="(c) Numerical variables")
plot(pca,choice="sqload",coloring.var=T, leg=TRUE,
     posleg="topright", main="(d) All variables")
par(mfrow=c(1,1))

pca <- PCAmix(X.quanti = scale(df.quanti), X.quali = df.quali, rename.level = TRUE, ndim = 20,graph=FALSE)
pcarot <- PCArot(pca,dim=17,graph=FALSE)
cutoff <- function(v, cutoffval = 0.4) {
  ifelse(v >= cutoffval, v, NA)
}
dat <- round(pcarot$sqload %>% as_tibble() %>% mutate_all(cutoff),2) %>% as.data.frame()
rownames(dat) <- rownames(pcarot$sqload)
DT::datatable(dat, options = list(scrollX = TRUE))

# Modeling

pacman::p_load(tidyverse, psych, gridExtra, broom, PCAmixdata, caret, scorecard,pROC, ROCR, DMwR, MASS, mob)

df <- as_tibble(read.csv('BankChurners.csv', header = T))
df <- df %>%
  dplyr::select(-Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1, -Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2, -CLIENTNUM, -Avg_Open_To_Buy)
df <- df %>%
  mutate_if(is.character, as.factor)

set.seed(123456)
index <- caret::createDataPartition(df$Attrition_Flag, p = 0.7, list = FALSE)
train_dat <- df[index,]
test_dat <- df[-index,]

prop.table(table(df$Attrition_Flag))
prop.table(table(train_dat$Attrition_Flag))
prop.table(table(test_dat$Attrition_Flag))

logit_full <- glm(Attrition_Flag~ ., family = binomial(link = "logit"), data = train_dat)
logit_stepwise<- stepAIC(logit_full, k=qchisq(0.05, 1, lower.tail=F), direction = "both",trace = F)
vif(logit_stepwise, merge_coef = TRUE)
DT::datatable(tidy((logit_stepwise)))
set.seed(123)
ctrl <- trainControl(method="cv", number = 10) 
tuneGrid <- expand.grid(kmax = 3:7,                        # test a range of k values 3 to 7
                        kernel = c("rectangular", "cos"),  # regular and cosine-based distance functions
                        distance = 1:2)                    # powers of Minkowski 1 to 3
temp <- train_dat
knnFit <- train(Attrition_Flag ~ ., 
                data = temp, 
                method = 'knn',
                trControl = ctrl,
                preProcess = c('center', 'scale'),
                tuneLength=15)
plot(knnFit)
table(train_dat$Attrition_Flag)
balanced.data <- SMOTE(Attrition_Flag~., as.data.frame(train_dat),perc.over = 100, k = 25)
table(balanced.data$Attrition_Flag)
logit_full_SMOTE <- glm(Attrition_Flag~ ., family = binomial(link = "logit"), data = balanced.data)
logit_stepwise_SMOTE<- stepAIC(logit_full_SMOTE, k=qchisq(0.05, 1, lower.tail=F), direction = "both",trace = F)
vif(logit_stepwise_SMOTE, merge_coef = TRUE)
DT::datatable(tidy((logit_stepwise_SMOTE)))
bins <- woebin(train_dat,y = "Attrition_Flag",count_distr_limit = 0.05, bin_num_limit = 10, positive = 'bad|Attrited Customer')

iv <- map_df(bins, ~pluck(.x, 10, 1)) %>%
  pivot_longer(everything(), names_to = "var", values_to = "iv") %>%
  mutate(group = case_when(iv < 0.02 ~ "Generally unpredictive",
                           iv < 0.1 ~ "Weak",
                           iv < 0.3 ~ "Medium",
                           iv <= 0.5 ~ "Strong",
                           TRUE ~ "Suspicious for overpredicting"))
iv %>%
  ggplot(aes(x = var, y = iv, fill = group)) +
  geom_col() +
  theme(text = element_text(size = 5)) +
  #scale_x_discrete(labels = abbreviate) +
  theme_minimal() +
  coord_flip()

woe_transform_df <- bins[[1]]
for (i in 2:length(bins)){
  woe_transform_df <- rbind(woe_transform_df,bins[[i]])
}

woe_transform_df %>%
  ggplot(aes(x = (bin), y = woe, group = 1)) +
  geom_line() +
  facet_wrap(~ variable) +
  theme_minimal() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())


non_linear <- c('Avg_Utilization_Ratio', 'Credit_Limit', 'Total_Revolving_Bal', 'Total_Trans_Amt')
dt <- df %>%
  dplyr::select(non_linear, Attrition_Flag) %>%
  mutate(Attrition_Flag = ifelse(Attrition_Flag == 'Attrited Customer', 1, 0) ) %>%
  as.data.frame()
result <- list()
for (j in non_linear) {
  result[[j]] <- iso_bin(x = dt[,j], y = dt$Attrition_Flag)$cut
}
breaks_adj <- woebin(train_dat, y = "Attrition_Flag", breaks_list=result,
                     count_distr_limit = 0.05, bin_num_limit = 10, positive = 'bad|Attrited Customer')
woe_transform_df_new <- breaks_adj[[1]]
for (i in 2:length(breaks_adj)){ 
  woe_transform_df_new <- rbind(woe_transform_df_new,breaks_adj[[i]])
}

woe_transform_df_new %>%
  ggplot(aes(x = (bin), y = woe, group = 1)) +
  geom_line() +
  facet_wrap(~ variable) +
  theme_minimal() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

iv_new <- map_df(breaks_adj, ~pluck(.x, 10, 1)) %>%
  pivot_longer(everything(), names_to = "var", values_to = "iv") %>%
  mutate(group = case_when(iv < 0.02 ~ "Generally unpredictive",
                           iv < 0.1 ~ "Weak",
                           iv < 0.3 ~ "Medium",
                           iv <= 0.5 ~ "Strong",
                           TRUE ~ "Suspicious for overpredicting"))

iv_new <- iv_new %>%
  filter(!group %in% c("Generally unpredictive"))

train_dat_WOE <- train_dat %>%
  dplyr::select(one_of(unique(iv_new$var)),Attrition_Flag)
test_dat_WOE <- test_dat %>%
  dplyr::select(one_of(unique(iv_new$var)),Attrition_Flag)

train_woe_list <- woebin_ply(train_dat_WOE, breaks_adj, to = "woe")
test_woe_list <- woebin_ply(test_dat_WOE, breaks_adj, to = "woe")

logit_full_WOE<- glm(Attrition_Flag~ ., family = binomial(link = "logit"), data = train_woe_list)
logit_stepwise_WOE <- stepAIC(logit_full_WOE, k=qchisq(0.05, 1, lower.tail=F), direction = "both",trace = F)
DT::datatable(tidy((logit_stepwise_WOE)))

logit_WOE <- glm(Attrition_Flag~ ., family = binomial(link = "logit"), data = train_woe_list %>% dplyr::select(-Total_Revolving_Bal_woe, -Avg_Utilization_Ratio_woe, -Customer_Age_woe))
DT::datatable(tidy((logit_WOE)))

df.quanti <- train_dat %>% dplyr::select(-where(is.factor)) %>% mutate_all(as.numeric) %>% as.data.frame()
df.quali <- train_dat %>% dplyr::select(where(is.factor),-Attrition_Flag) %>% as.data.frame()
pca <- PCAmix(X.quanti = scale(df.quanti), X.quali = df.quali, rename.level = TRUE, ndim = 17, graph = FALSE)
temp <- as.data.frame(pca$ind$coord)
temp$Attrition_Flag <- train_dat$Attrition_Flag
logit_full_PCA <- glm(Attrition_Flag~ ., family = binomial(link = "logit"), data = temp)
logit_stepwise_PCA <- stepAIC(logit_full_PCA, k=qchisq(0.05, 1, lower.tail=F), direction = "both",trace = F)
DT::datatable(tidy((logit_stepwise_PCA)))

test_dat$BENCHMARK <- predict(logit_stepwise, newdata = test_dat, type = "response")
test_dat$SMOTE <- predict(logit_stepwise_SMOTE, newdata = test_dat, type = "response")
test_dat$WOE <- predict(logit_WOE, newdata = test_woe_list, type = "response")
df.quanti.test <- test_dat %>% dplyr::select(-where(is.factor),-BENCHMARK, -SMOTE,-WOE) %>% mutate_all(as.numeric) %>% as.data.frame()
df.quanti.test <- test_dat %>% dplyr::select(-where(is.factor),-BENCHMARK, -SMOTE,-WOE) %>% mutate_all(as.numeric) %>% as.data.frame()
df.quali.test <- test_dat %>% dplyr::select(where(is.factor),-Attrition_Flag) %>% as.data.frame()
test_temp <- as.data.frame(predict(pca,X.quanti = scale(df.quanti.test), X.quali = df.quali.test,rename.level = TRUE, graph = FALSE))
colnames(test_temp) <- paste('dim', 1:17)
test_dat$PCA <- predict(logit_stepwise_PCA, newdata = (test_temp), type = "response")

par(mfrow = c(2,2))
plot(roc(test_dat$Attrition_Flag, test_dat$BENCHMARK, direction="<"), col="steelblue", lwd=3, main="(a) Benchmark",print.auc=TRUE)
plot(roc(test_dat$Attrition_Flag, test_dat$SMOTE, direction="<"), col="steelblue", lwd=3, main="(b) SMOTE technique",print.auc=TRUE)
plot(roc(test_dat$Attrition_Flag, test_dat$WOE, direction="<"), col="steelblue", lwd=3, main="(c) WOE technique",print.auc=TRUE)
plot(roc(test_dat$Attrition_Flag, test_dat$PCA, direction="<"), col="steelblue", lwd=3, main="(d) PCA technique",print.auc=TRUE)
par(mfrow = c(1,1))

roc_step <- roc(test_dat$Attrition_Flag, test_dat$BENCHMARK, direction="<")
d <-coords(roc_step,"best","threshold",transpose=T)
cm_benchmark <- confusionMatrix(as.factor(ifelse(test_dat$BENCHMARK >= d[[1]],"Existing Customer", "Attrited Customer")), test_dat$Attrition_Flag)
recall_bm <- cm_benchmark$byClass[1] 
roc_step <- roc(test_dat$Attrition_Flag, test_dat$SMOTE, direction="<")
d <-coords(roc_step,"best","threshold",transpose=T)
cm_SMOTE <- confusionMatrix(as.factor(ifelse(test_dat$SMOTE >= d[[1]],"Existing Customer", "Attrited Customer")), test_dat$Attrition_Flag)
recall_SMOTE <- cm_SMOTE$byClass[1] 
roc_step <- roc(test_dat$Attrition_Flag, test_dat$WOE, direction="<")
d <-coords(roc_step,"best","threshold",transpose=T)
cm_WOE <- confusionMatrix(as.factor(ifelse(test_dat$WOE >= d[[1]],"Existing Customer", "Attrited Customer")), test_dat$Attrition_Flag)
recall_WOE <- cm_WOE$byClass[1] 
roc_step <- roc(test_dat$Attrition_Flag, test_dat$PCA, direction="<")
d <-coords(roc_step,"best","threshold",transpose=T)
cm_PCA <- confusionMatrix(as.factor(ifelse(test_dat$PCA >= d[[1]],"Existing Customer", "Attrited Customer")), test_dat$Attrition_Flag)
recall_PCA <- cm_PCA$byClass[1] 

rbind(recall_bm,recall_SMOTE,recall_WOE,recall_PCA) %>%
  as.data.frame() %>%
  mutate(Model = c("Benchmark","SMOTE","WOE","PCA")) %>%
  ggplot() +
  geom_col(aes(x = Model, y = Sensitivity), fill = 'steelblue') +
  geom_text(aes(x = Model,y = Sensitivity, label = round(Sensitivity,3)), vjust = -0.5) +
  theme_minimal() +
  ylab('Recall')


