data <- read.delim("/Users/kevindu/Desktop/Coding/Tests:experiments/MarkovGame/log.out", sep=',', header = FALSE)

plot(1:length(data), log(data), type='l')