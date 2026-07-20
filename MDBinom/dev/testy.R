# TODO: Add comment
# 
# Author: Piotr
###############################################################################

n <- 1000
x <- rnorm(n)
y <- rbinom(n, size=1, prob=0.3)
reg_nieparam(x,y)

#czy b³¹d dla y=NA
reg_nieparam(c(x,1),c(y,NA))

#czy b³¹d dla x=NA
reg_nieparam(score=c(NA, x,NA),default=c(3,y,1), buckets=2, na.omit=FALSE)

buckety_br(c(NA, x,NA),c(3,y,1), 20, method = "eq_count")




setwd('G:\\Michal\\eclipse - java\\MDbinom')
library(lineprof)
source("testy_prof.R")
l <- lineprof(f(),interval = 0.000001)
shine(l)



wynik<-univariate_loop(x_df=dane_tr, y=dane_tr$churn2_8)

genRaport(wynik, dir="c:/temp/raport")



univariate_anal_stats1(x=c(2,2,2,2), y=c(1,0,0,0))

univariate_anal_stats(x=c(2,2,2,2), y=c(1,0,0,0))

univariate_anal_stats(x=dane_tr$Nazwisko_Nazwa, y=dane_tr$churn2_8)

wyniki<-univariate_loop(dane_tr, dane_tr$churn2_8)
genRaport(wyniki, dir="c:/temp/raport")


wynik<-univariate_anal_stats(dane_tr$Wiek, dane_tr$churn2_8, czas=dane_tr$miesiacZamowienia)
windows()
genRaport(list('a'=wynik))

drzewo_plot(wynik$dyskretyzacja, xlab = 'zmienna', ylab = "Average target", 
		main = paste('zmienna', "discretization"))


windows()
reg_nieparam(score=dane_tr$Wiek, default=dane_tr$churn2_8)


model<-glm(churn2_8~Wiek, data=dane_tr, family='binomial')
pred<-predict(model, type='response')

dopasowanie_do_zmiennej(dane_tr$Wiek, pred, wynik[[1]])


make_model_formula("daa",c("a","b"))

step_bez_kor(data=dane_tr[,c("AboCenaPodstawowa","ModemCenaPodstawowa")], model=model, target_var_name='churn2_8')



iks<-function(t, l=1/1000, vw=100, vs=100){
	print(-vs/vw*log(l)*(l+vw*t) + (l+vw*t)*vs/vw*log(l+vw*t))
	print(vw*t+l)
}

iks(t=1, l=100, vw=100, vs=144.27)


curve(iks, from=0, to=100, n=100)




N<-60*60*10
czas<-seq(from=0, to = 5/60/60, length.out=N)

x_wozu <- rep(NULL, N)
v_gumy <- rep(NULL, N)
v_razem <- rep(NULL, N)
x_slimaka <- rep(NULL, N)
dx_slimaka <- rep(NULL, N)

l<-0.0001
x_slimaka[1] <-  0
v_slimaka <- 1
v_wozu <- 100
dt=czas[2]-czas[1]

for (i in 1:(N-1)){
	t<-czas[i]
	x_wozu[i] <- v_wozu*t+l
	v_gumy[i] <- x_slimaka[i]/x_wozu[i]*v_wozu
	v_razem[i] <- v_slimaka + v_gumy[i]
	dx_slimaka[i] <- v_razem[i] * dt
	x_slimaka[i+1]<-x_slimaka[i]+dx_slimaka[i]
}
tail(x_slimaka)


df<-data.frame(czas[-1], x_wozu, v_gumy, v_slimaka, v_razem, dx_slimaka, x_slimaka)

edit()


