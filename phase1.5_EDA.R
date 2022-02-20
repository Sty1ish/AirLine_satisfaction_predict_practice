train = read.csv('C:/Users/Nyoths/Desktop/프로젝트 관련/동아리 2월 pj - 항공사/train.csv')
library(ggplot2)
ggplot( train, aes(x=Departure.Delay.in.Minutes)) + geom_histogram(fill="#F8766D", colour="black", bins = 30) + scale_x_continuous(limits = c(0, 300))
ggplot( train, aes(x=Arrival.Delay.in.Minutes)) + geom_histogram(fill="#F8766D", colour="black", bins = 30) + scale_x_continuous(limits = c(0, 300))
