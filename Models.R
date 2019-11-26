install.packages("readxl")
install.packages('car')
library("readxl")
library('car')

#Data Cleaning was performed in Python and hence the dataset is different from
#actual non clean data. Please use the file provided along with submission.

getwd()
setwd('C:/Users/CosmicDust/Documents/Statistics/Assignment 5/Submission')

df = read_excel("DIA-Clean Data.xlsx")

df = as.data.frame(df)

names(df)
summary(df)

# Creating Independent Variables
df$ED = (df$Enplaned+(log(df$Deplaned)))
df$OD = (df$Originating - df$Transfer + (log(df$Destination)))
A = log(df$Parking)
B = df$RentalCar*df$RentalCar*df$RentalCar
C = df$Ground*df$Ground
df$PGR = ((A+B-C)/df$Parking)*(A+B-C)/df$Parking
df$G = (df$Ground*df$Ground)/log(df$OriginDestin)
D = df$Concession*df$Concession
df$CPG = ((A+B-D)/df$Parking)*(A+B-D)/df$Parking
df$EDL = df$Enplaned-log(df$Deplaned)
df$CP = df$Concession - log(df$Parking)
df$UOD = (df$UMCSENT*log(df$OriginDestin))/df$OriginDestin


# Creating Train and Test Data
train = df[c(1:62),]
test = df[c(63:66),]
train_UMCSENT = df[c(2:62),]

#--------------------------------CONCESSION-------------------------------------

# Create the relationship model.
model <- lm(Concession ~ ED + OD + (PGR) + Month, data = train)
predict(model, newdata = test)

# Show the model.
summary(model)
vif(model)
anova(model)
AIC(model)
BIC(model)
confint(model)

names(df)

# error
s_actval = sum(df[c(63:66),8])
s_predval = sum(predict(model, newdata = test))

error_Concession = ((s_actval - s_predval)/s_actval)*100

# Print error
print(error_Concession)

# Checking Assumptions for linear model
layout(matrix(c(1,2,3,4),2,2,byrow=T))
plot(model$fitted, rstudent(model),
     main="Multi Fit Studentized Residuals",
     xlab="Predictions",ylab="Studentized Resid",
     ylim=c(-2.5,2.5))
abline(h=0, lty=2)
plot(model$fitted.values,model$residuals,main='Fitted values v Residuals')
abline(h=0,lty=2)
hist(model$resid,main="Histogram of Residuals")
qqnorm(model$resid) + qqline(model$resid)

plot(model)


#--------------------------------Parking-------------------------------------

# Create the relationship model.
model <- lm(Parking ~ ED + OD + G + Month , data = train)
predict(model, newdata = test)

# Show the model.
summary(model)
vif(model)
anova(model)
AIC(model)
BIC(model)

names(df)

# error
s_actval = sum(df[c(63:66),9])
s_predval = sum(predict(model, newdata = test))

error_Parking = ((s_actval - s_predval)/s_actval)*100

# Print error
print(error_Parking)

# Checking Assumptions for linear model
layout(matrix(c(1,2,3,4),2,2,byrow=T))
plot(model$fitted, rstudent(model),
     main="Multi Fit Studentized Residuals",
     xlab="Predictions",ylab="Studentized Resid",
     ylim=c(-2.5,2.5))
abline(h=0, lty=2)
plot(model$fitted.values,model$residuals,main='Fitted values v Residuals')
abline(h=0,lty=2)
hist(model$resid,main="Histogram of Residuals")
qqnorm(model$resid) + qqline(model$resid)

plot(model)


#--------------------------------RENTAL CAR-------------------------------------

# Create the relationship model.
model <- lm(RentalCar ~ ED + OD + Month + Cannabis, data = train)
predict(model, newdata = test)

# Show the model.
summary(model)
vif(model)
anova(model)
AIC(model)
BIC(model)

# error
s_actval = sum(df[c(63:66),10])
s_predval = sum(predict(model, newdata = test))

error_RentalCar = ((s_actval - s_predval)/s_actval)*100

# Print error
print(error_RentalCar)

# Checking Assumptions for linear model
layout(matrix(c(1,2,3,4),2,2,byrow=T))
plot(model$fitted, rstudent(model),
     main="Multi Fit Studentized Residuals",
     xlab="Predictions",ylab="Studentized Resid",
     ylim=c(-2.5,2.5))
abline(h=0, lty=2)
plot(model$fitted.values,model$residuals,main='Fitted values v Residuals')
abline(h=0,lty=2)
hist(model$resid,main="Histogram of Residuals")
qqnorm(model$resid) + qqline(model$resid)

plot(model)

#--------------------------------GROUND-------------------------------------

# Create the relationship model.
model <- lm(Ground ~ OD + EDL + CP + Month, data = train)
predict(model, newdata = test)


# Show the model.
summary(model)
vif(model)
anova(model)
AIC(model)
BIC(model)


# Calculate error
s_actval = sum(df[c(63:66),11])
s_predval = sum(predict(model, newdata = test))

error_Ground = ((s_actval - s_predval)/s_actval)*100

# Print error
print(error_Ground)

# Checking Assumptions for linear model
layout(matrix(c(1,2,3,4),2,2,byrow=T))
plot(model$fitted, rstudent(model),
     main="Multi Fit Studentized Residuals",
     xlab="Predictions",ylab="Studentized Resid",
     ylim=c(-2.5,2.5))
abline(h=0, lty=2)
plot(model$fitted.values,model$residuals,main='Fitted values v Residuals')
abline(h=0,lty=2)
hist(model$resid,main="Histogram of Residuals")
qqnorm(model$resid) + qqline(model$resid)

plot(model)


#--------------------ALTERNATE MODELS WITH UMCSENT VALUES-----------------------


#--------------------------------Parking----------------------------------------

# Create the relationship model.
model <- lm(Parking ~ ED + OD + G + Month + UMCSENTLag1 , data = train_UMCSENT)
predict(model, newdata = test)

# Show the model.
summary(model)
vif(model)
anova(model)
AIC(model)
BIC(model)


# error
s_actval = sum(df[c(63:66),11])
s_predval = sum(predict(model, newdata = test))

error_Parking = ((s_actval - s_predval)/s_actval)*100

# Print error
print(error_Parking)

# Checking Assumptions for linear model
layout(matrix(c(1,2,3,4),2,2,byrow=T))
plot(model$fitted, rstudent(model),
     main="Multi Fit Studentized Residuals",
     xlab="Predictions",ylab="Studentized Resid",
     ylim=c(-2.5,2.5))
abline(h=0, lty=2)
plot(model$fitted.values,model$residuals,main='Fitted values v Residuals')
abline(h=0,lty=2)
hist(model$resid,main="Histogram of Residuals")
qqnorm(model$resid) + qqline(model$resid)

plot(model)


#--------------------------------RENTAL CAR-------------------------------------

# Create the relationship model.
model <- lm(RentalCar ~ ED + OD + Month + Cannabis + UMCSENTLag1 , data = train_UMCSENT)
predict(model, newdata = test)

# Show the model.
summary(model)
vif(model)
anova(model)
AIC(model)
BIC(model)

# error
s_actval = sum(df[c(63:66),12])
s_predval = sum(predict(model, newdata = test))

error_RentalCar = ((s_actval - s_predval)/s_actval)*100

# Print error
print(error_RentalCar)

# Checking Assumptions for linear model
layout(matrix(c(1,2,3,4),2,2,byrow=T))
plot(model$fitted, rstudent(model),
     main="Multi Fit Studentized Residuals",
     xlab="Predictions",ylab="Studentized Resid",
     ylim=c(-2.5,2.5))
abline(h=0, lty=2)
plot(model$fitted.values,model$residuals,main='Fitted values v Residuals')
abline(h=0,lty=2)
hist(model$resid,main="Histogram of Residuals")
qqnorm(model$resid) + qqline(model$resid)

plot(model)

#--------------------------------GROUND-------------------------------------

# This models error output and r^2 is better but VIF is higher. Thants why, we did not choose this model.

# Create the relationship model.
model <- lm(Ground ~ OD + EDL + CP + Month + log(UMCSENTLag1) , data = train_UMCSENT)
predict(model, newdata = test)


# Show the model.
summary(model)
vif(model)
anova(model)
AIC(model)
BIC(model)


# Calculate error
s_actval = sum(df[c(63:66),13])
s_predval = sum(predict(model, newdata = test))

error_Ground = ((s_actval - s_predval)/s_actval)*100

# Print error
print(error_Ground)

# Checking Assumptions for linear model
layout(matrix(c(1,2,3,4),2,2,byrow=T))
plot(model$fitted, rstudent(model),
     main="Multi Fit Studentized Residuals",
     xlab="Predictions",ylab="Studentized Resid",
     ylim=c(-2.5,2.5))
abline(h=0, lty=2)
plot(model$fitted.values,model$residuals,main='Fitted values v Residuals')
abline(h=0,lty=2)
hist(model$resid,main="Histogram of Residuals")
qqnorm(model$resid) + qqline(model$resid)

plot(model)