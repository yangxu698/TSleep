getwd()
setwd("../Tesserae")
library(dplyr)

data0 = read.csv("derived_features_9_23.csv")
str(data0)
summary(data0)
## data0.noNA = na.omit(data0)

colnames = sort(colnames(data0))
data1 = data0 %>% select(AcuteStressPrior2HR, AcuteStressRelativeToDay, AcuteStressRelativeToWeek,
                         EpisodicStressDayRelativeToWeek, EpisodicStressDayRelativeToMonth,
                         LifetimeAvgHR, AverageCurrentHR, LifetimeAvgHRV, HrsSleep, CurrentSteps,
                         CurrentExercizeMins, TimeOfDay, WorkDay, SurveyTime)
write.csv(colnames, "colnames.csv")
samples10 = data0 %>% sample_n(10) %>% select(garmin_sleep_duration, adjusted_sleep_duration,
                                              bed_time, wakeup_time, daily_sleep_debt, daily_sleep_debt_adjusted,
                                              survey_completion_time_from_start, survey_end_datetime,
                                              survey_start_datetime)
samples10 = data0 %>% sample_n(10) %>% select(sort(names(.)))
samples10$current_hr_sent_time
write.csv(t(samples10),"samples10.csv")
glimpse(data0)
