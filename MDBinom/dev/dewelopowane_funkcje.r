

library('Hmisc')
library('MDBinom')

#########


load("C:\\Users\\Piotr\\workspace\\CP\\dane02.RData")
load("C:\\Users\\Piotr\\workspace\\CP\\wyniki_univ.RData") 

######

###########################################  interakcje ##########################################

lista_zmiennych<-c('AboCenaPodstawowa','Kara','WhiteListPESEL','dataAktNajstarszegoKontr','Wiek')

zmienne_raport<-univariate_loop(dane, dane$target, 
		vsub_list = lista_zmiennych)
genRaport(zmienne_dyskr, show=TRUE, )


zmienne_dyskr <- lapply(zmienne_raport, function(x)x[[1]])


dane_nowe_zmienne<-przypisz_z_listy(bucket_list = zmienne_dyskr
						, data = dane
						, vars = lista_zmiennych
						, colname='label'
						, varname_sufix='label')
				

				
target<-dane$target
var1<-dane_nowe_zmienne$AboCenaPodstawowa_label
var2<-dane_nowe_zmienne$Wiek_label

interactionsTest(var1,var2, target)

wynik<-interactionsTestAll(dane
		, zmienne_dyskr)

		, vars_names=names(discr_list)
		, target=dane$target
		, colname='label'
		, varname_sufix='label')


data<-dane
discr_list <- zmienne_dyskr
vars_names=names(discr_list)
target=dane$target
colname='label'
varname_sufix='label'








windows()
interakcja(var1, var2, target)
interakcja(var2, var1, target)


##################################################################################################



buckety_stat_wtd<-function(bucket, default, weights=rep(1,length(bucket)), total=TRUE)
{

#	dane<-as.data.frame(cbind(score, default));

	#usuwam puste levele factorów
	if (is.factor(bucket))
		bucket<-factor(bucket);

	obs<- wtd.table(bucket, weights=weights);
	obs_all<-sum(weights);
	bad<- as.vector(tapply(default, bucket, wtd.sum, weights=weights));
	bad_all<-wtd.sum(bad, weights=weights);
	br<- as.vector(tapply(default, bucket, wtd.mean, , weights=weights));
	std_dev<-sqrt(as.vector(tapply(default, bucket, wtd.var, weights=weights)));
	logit<-log(br/(1-br));
	probit<-qnorm(br);

	#do_wagi<-bad;
	#do_wagi[do_wagi==0]<-0.5;
	#do_wagi[do_wagi==obs]<-obs[do_wagi==obs]-0.5;
	#waga<-obs/((do_wagi)/(obs)*(1-(do_wagi)/(obs)));
	waga<-NA;

	pct_good=(obs-bad)/(obs_all-bad_all);
	pct_bad=bad/bad_all;
	woe=log(pct_good/pct_bad)

	wynik<-as.data.frame(cbind( nr=1:length(obs),
											 n_good=(obs-bad), pct_good,
									     n_bad=bad, pct_bad,
											 n_obs=obs, pct_obs=obs/obs_all,
											 gb_odds=(obs-bad)/bad,
										   br, std_dev, woe, logit, probit, var=br*(1-br)/obs, waga))

  rownames(wynik)<-sort(unique(bucket));

	if (total) {
			wynik<-rbind(wynik, TOTAL=c(length(obs)+1, obs_all-bad_all, 1, bad_all, 1, obs_all,1,
														(obs_all-bad_all)/bad_all, bad_all/obs_all, sd(default), 0, NA, NA, NA, NA));
  }
	return(wynik);
}


n<-1000;
x<-rnorm(n);
y<-as.numeric(x+rnorm(n)<0);

drzewo(x, y)
traceback()



wtd.reg_nieparam<-function (score, default, buckets = 100, subset=NULL, wytnij = 0, span = 0.7,
		degree = 2, plot = TRUE, target = "br", new = TRUE, col_points = "black",
		col_line = "darkblue", index = FALSE, weights=NULL, estim=NULL, ...)
{
	dane <- data.frame(score, default)
	
	if (is.null(weights)){
		weights=rep(1,length(score))
		weighst_locfit=weights	
	}
	else{
		
		# TODO zmieniæ to!
		# na cele tego zadania przyhardkorzy³em i obchodzê b³¹d locfita w ten sposób,
		# ¿e ustalam wagê 1 dla obserwacji z target=1
		
		waga_1<-weights[which(default==1)[1]]
		weights_locfit<-weights/waga_1
	}
	
	#jeœli okreœlono subset, to ograniczam dane na których pracujemy 
	if (!is.null(subset)){
		dane<-dane[subset,]
		weights<-weigths[subset]
		estim<-estim[subset]
	}
	
	if (wytnij > 0){
		do_usuniecia<-usun_konce(dane$score, prob = wytnij, weights=weights);
		if (length(do_usuniecia)>0)
			dane <- dane[-do_usuniecia,]
	}
	bucket <- buckety_br(x=dane$score, y=dane$default, n=buckets, method = "eq_count", weights=weights)
	
	
	
	#jeœli s¹ dwie wartoœci y, to uznaje to jest to zmienna binarna, i stosuje regresjê logistyczn¹
	if (length(unique(default)) == 2)
		l <- locfit(default ~ lp(score, nn = span), family = "binomial",
				link = "logit", data = dane, weights=weights_locfit)
	#w przeciwnym razie robi regresje liniow¹
	else 
		l <- locfit(default ~ lp(score, nn = span), data = dane, weights=weights)
	
	b2 <- predict(l, newdata = bucket$srodek)
	if (target == "br")
		bucket2 <- cbind(bucket, fitted = b2)
	else bucket2 <- cbind(bucket, fitted = log(b2/(1 - b2)))
	
	#liczê wielkoœæ b¹belka
	skala <- sqrt(bucket$n_obs/(sum(weights)/buckets))
	
	#liczê wartoœci wyestymowane
	#estim_aggr<-buckety_stat(b2, default, )
	
	#rysowanie
	x <- bucket2$srodek
	if (index)
		x <- bucket$nr
	if (plot) {
		if (new == TRUE)
			plot(x, with(bucket2, get(target)), col = col_points,
					cex = skala, ...)
		else points(x, with(bucket2, get(target)), cex = skala,
					col = col_points, ...)
		lines(x, bucket2$fitted, col = col_line, ...)
	}
	bucket2
}
