# Funkcje zaczynające się na univariate
# 
# Author: Michał Danaj
###############################################################################



#' Dyskretyzuje zmienną i wylicza dla niej statystyki.
#'
#' Wylicza statystyki i zwraca je w postaci listy.
#' @param x zmienna, po której procedura będzie sortować.
#' @param y zmienna odpowiedzi.
#' @param czas podział na okresy czasowe.
#' @param proby \code{data.frame}, w którym każda kolumna jest wektorem logicznym,
#'       zawierającym informacje, czy obserwacja należy do danej próby.
#' @param interactive TRUE, jeśli zmienna ma być dyskretyzowana interaktywnie. W
#'                   przeciwnym razie, co jest wartością domyślną, dyskretyzacja
#'                   jest automatyczna.
#' @param min_bucket minimalna liczba obserwacji w buckecie, przy dzieleniu drzewem.
#' @param breaks zamiast automatycznego dzielenia, można podać wartości przedziałów (from,to].
#' @param mapping zamiast automatycznego dzielenia, można podać mapowanie.
#' @param forceContinous wymusza potraktowanie zmiennej jako ciągłą, mimo że liczba
#'                      unikalnych wartości jest mniejsza niż \code{discrete_threshold}.
#' @param special_val Wartości specjalne do usunięcia z automatycznego podziału. Będą traktowane jako zmienne
#' 					  kategoryczne.
#' @param NA_subst wartość jaka ma być przypisana w miejsce braków danych. Dalsze analizy
#'		będą przeporwadzone w standardowy sposób. Jeśli jednak wartość \code{NA_subst} zostanie
#' 		dodana do \code{special_val}, zostanie ona potraktowana jako wartość dyskretna.
#' @param span Parametr wygładzający funkcji \code{\link[locfit]{locfit}}.  
#' @author Michał Danaj
#' @export
univariate_anal_stats<-function(x,y,czas,proby=rep(TRUE, length(y)),
		interactive=FALSE, min_bucket=floor(0.05*length(x)), breaks=NULL,
		mapping=NULL, forceContinous=FALSE, 
		special_val=numeric_var_treatment.params$special_val, 
		NA_subst=numeric_var_treatment.params$NA_substit,
		span=0.9){
	
	# zamieniam braki danych na liczbę.
	#if (is.numeric(x) & !is.null(NA_subst))
	#	x[is.na(x)]<-NA_subst;
	
	
	# dyskretyzuję zmienną i wyliczam pierwsze statystyki
	stat1<-univariate_anal_stats1(x,y, special_val=special_val,
			max_gleb=3,plot=FALSE, min_bucket=min_bucket,
			interactive=interactive, breaks=breaks, mapping=mapping,
			forceContinous=forceContinous, span=span);
	
	stat2<-NULL;
	stat3<-NULL;
	#Dalsze statystyki robię pod warunkiem, że jest więcej niż jedna wartość dyskretna
	#(W stat1 jest też <TOTAL>, dlatego 2)
	if (typeof(stat1)!="character"){
		
		# przypisuję nazwę bucketu
		stat1$fitted<-stat1$label;
		x_discr<-przypisz2(x,stat1);
		
		# przypisuję BR
		stat1$fitted<-stat1$br;
		BR_discr<-przypisz2(x,stat1);
		
		# wyliczam drugie statystyki
		stat2<-univariate_anal_stats2(x_discr, y, czas, BR_discr);
		
		# wyliczam trzecie statystyki (GINI) po zadanych próbach i czasie
		stat3<-univariate_anal_stats3(score=-BR_discr, y, czas, proby);
		
	}
	return(list(dyskretyzacja=stat1, rozklady=stat2, dyskryminacja=stat3));
}



####################




################################################################################
#                    zmienione funkcjie z pakietu binom                        #
################################################################################





#' Wylicza statystyki dla zmiennej score'owej i objaśnianej wg zadanych bucketów
#'
#' Wylicza statystyki i zwraca je w postaci listy. Do wyliczenia \code{AR} używa jako zmiennej
#' score'owej oryginalnych wartości BR przypisanych do bucketu. Tzn, mimo że na nowych danych kolejność
#' bucketów, sortując je po br może być inna niż oryginalnie, to stosowana jest oryginalna kolejność.
#' @param buckets \code{data.frame} z podziałem zmiennej na buckety.
#' @param x zmienna, po której procedura będzie sortować.
#' @param y zmienna odpowiedzi.
#' @param czas podział na okresy czasowe.
#' @param proby \code{data.frame}, w którym każda kolumna jest wektorem logicznym,
#'       zawierającym informacje, czy obserwacja należy do danej próby.
#' @author Michał Danaj
#' @export
univariate_stats_new_data<-function(buckets,x,y,czas,proby=rep(TRUE, length(y))){
	buckets$fitted<-buckets$label;
	nowe_wartosci<-przypisz2(x, buckets)
	buckets_new<-univariate_anal_stats(nowe_wartosci, y, czas=czas, proby=proby)
	stat1<-cbind(buckets[,c('nr','label','discret','od','srodek','do')], 
			buckets_new$dyskretyzacja[buckets$label,c('n_good','pct_good','n_bad','pct_bad','n_obs','pct_obs','br','woe','logit')],
			br_orig=buckets$br);
	
	stat2<-NULL;
	stat3<-NULL;
	#Dalsze statystyki robię pod warunkiem, że jest więcej niż jedna wartość dyskretna
	#(W stat1 jest też <TOTAL>, dlatego 2)
	if (nrow(stat1)>2){
		# przypisuję nazwę bucketu
		stat1$fitted<-stat1$label;
		x_discr<-przypisz2(x,stat1);
		
		# przypisuję BR
		stat1$fitted<-stat1$br;
		BR_discr<-przypisz2(x,stat1);
		
		# wyliczam drugie statystyki
		stat2<-univariate_anal_stats2(x_discr, y, czas, BR_discr);
		
		# wyliczam trzecie statystyki (GINI) po zadanych próbach i czasie
		buckets$fitted<-buckets$br;
		stare_BR<-przypisz2(x,buckets);
		stat3<-univariate_anal_stats3(-stare_BR, y, czas, proby);
		attr(stat3,'comment')<-"AR policzony w oparciu o oryginalne wartości BR."
	}
	return(list(dyskretyzacja=stat1, rozklady=stat2, dyskryminacja=stat3));
}


#' Robi rozkład zmiennej po czasie (lub innym podziale)
#'
#' Wylicza liczności dla każdego poziomu \code{x_discr} oraz średnią wartość \code{y} oraz \code{esitm}
#' w podziale na zadane grupy czasowe \code{czas} (lub podział innego typu).
#' @param x_discr zmienna objaśniająca.
#' @param y zmienna odpowiedzi.
#' @param czas Czas.
#' @param estim wartość wyestymowana przez model.
#' @export
univariate_anal_stats2<-function(x_discr, y, czas, estim){
	
	#liczności po czasie i zmiennej
	total<-table(x_discr);
	total_czas<-table(czas);
	all_tbl<-cbind(table(x_discr, czas), TOTAL=total);
	all_tbl<-rbind(all_tbl, c(total_czas, length(x_discr)));
	rownames(all_tbl)[nrow(all_tbl)]<-'TOTAL';
	
	
	#rozkład każdego bucketu po czasie (agregując po czasie zsumuje się do 1)
	pct_all_tbl<-sweep(all_tbl, 2, STATS=c(total_czas,length(x_discr)), FUN="/");
	
	#średnie LGD po czasie i zmiennej
	total<-tapply(y, list(x_discr), mean)
	avg_y<-tapply(y, list(x_discr,czas), mean);
	avg_y<-cbind(avg_y, TOTAL=total);
	total_czas<-c(tapply(y, list(czas), mean), mean(y));
	avg_y<-rbind(avg_y, total_czas);
	rownames(avg_y)[nrow(avg_y)]<-'TOTAL';
	return(list(obs_all_tbl = all_tbl, pct_all_tbl=pct_all_tbl, avg_t_tbl=avg_y,
					estim=tapply(estim, czas, mean)));
}

#' Wylicza AR po czasie i w podziale na zadane próby
#'
#' @param score zmienna, po której procedura będzie sortować.
#' @param y zmienna odpowiedzi.
#' @param czas podział na okresy czasowe.
#' @param proby \code{data.frame}, w którym każda kolumna jest wektorem logicznym,
#'       zawierającym informacje, czy obserwacja należy do danej próby.
#' @export
univariate_anal_stats3<-function (score, y, czas, proby){
	razem<-data.frame(score,y,czas);
	razem$czas<-as.factor(razem$czas);
	
	if (!is.data.frame(proby)){
		proby<-as.data.frame(proby);
	}
	
	
	for (i in 1:ncol(proby)){
		proba<-razem[proby[,i], ];
		AR_calosc<-AR(proba$score, proba$y)['AR'];
		aery<-sapply(split(proba, proba$czas), FUN=function(pr){
					
					if (nrow(pr)==0)
						return(NA);
					wyn<-AR(pr$score, pr$y)['AR'];
					names(wyn)<-NULL;
					return(wyn)
				}
		);
		
		if (i==1){
			wyniki_AR<-data.frame(aery);
			wyniki_AR<-cbind(AR_calosc,t(wyniki_AR));
		}
		else
			wyniki_AR<-rbind(wyniki_AR, c(AR_calosc,aery));
	}
	
	rownames(wyniki_AR)<-colnames(proby);
	colnames(wyniki_AR)<- colnames(wyniki_AR);
	return(wyniki_AR);
}



#' Dyskretyzuje zmienną i wylicza na niej statystyki
#'
#' W przypadku, gdy liczba unikalnych wartości zmiennej jest <= \code{discrete_threshold}
#' lub zmienna nie jest zmienną numeryczną,
#' uznaje że zmienna jest dyskretna i jedynie wylicza dla niej statystyki. W przeciwnym
#' wypadku dyskretyzuje zmienną i wylicza statystyki.
#' @param x zmienna, po której procedura będzie sortować.
#' @param y zmienna odpowiedzi.
#' @param locfit Czy z automatu dopasować funkcję z modelu \code{locfit}. 
#' @param discrete_threshold jeśli liczba unikalnych wartości zmiennej jest nie większa
#'        ta wartość, zmienna uznana jest za dyskretną i nie jest poddawana dyskretyzacji.
#' @param no_stats_threshold liczba unikalnych wartości zmiennej kategorycznej, powyżej której nie są generowane
#' 		  statystyki. W przypadku przekroczenia, zwracany jest komunikat "Too many categorical levels".
#' @param NA_substit wartość, którą zastąpić brak danych
#' @param special_val Wartości specjalne do usunięcia z automatycznego podziału. Będą traktowane jako zmienne
#' 					  kategoryczne.
#' @param max_gleb Maksymalna głębokośc do której budujemy drzewo
#' @param min_bucket Minimalna wielkość liścia
#' @param interactive TRUE, jeśli zmienna ma być dyskretyzowana interaktywnie. W
#'                   przeciwnym razie, co jest wartością domyślną, dyskretyzacja
#'                   jest automatyczna.
#' @param breaks zamiast automatycznego dzielenia, można podać wartości przedziałów (from,to].
#' @param mapping zamiast automatycznego dzielenia, można podać mapowanie.
#' @param forceContinous wymusza potraktowanie zmiennej jako ciągłą, mimo że liczba
#'                      unikalnych wartości jest mniejsza niż \code{discrete_threshold}.
#' @param span Parametr wygładzający funkcji \code{locit}.
#' @param ...  dodatkowe parametry graficzne.
#' @seealso \code{\link{buckety_stat}}.
#' @export
univariate_anal_stats1<-function(x,y, 
		locfit=FALSE, 
		discrete_threshold=numeric_var_treatment.params$discrete_threshold,
		NA_substit = numeric_var_treatment.params$NA_substit,
		special_val=numeric_var_treatment.params$special_val, 
		no_stats_threshold=numeric_var_treatment.params$no_stats_threshold,
		max_gleb=3, 
		min_bucket=200, 
		interactive=FALSE,
		breaks=NULL, 
		mapping=NULL, 
		forceContinous=FALSE,
		span=0.9,...){
	
		
	if (length(x)!=length(y))
		stop("paramet ry 'x' i 'y' mają różne długości!");
		
	
	## jeśli są jakieś nulle w x, to odpowiednio się nimi zajmuję
	nulle<-is.na(x)
	if (any(nulle)){
		
		#jeśli nulli jest mniej niż założona część populacji, to imputuję je. W przeciwnym razie przypisuję jako osobną grupę.
		ile_nulli<-prop.table(table(nulle))
		if (ile_nulli["TRUE"]<numeric_var_treatment.params$nulle_do_imp_thr)
			#TODO zobaczyć, czy y ma dwie wartości i jest to 0 i 1
			x<-missing_bin_target(x, y)
		else
			x[nulle]<-NA_substit;
		
	}
	
	
	## jeśli jest to zmienna dyskretna lub mapowanie
	if (!is.null(mapping)||((length(unique(x))<=discrete_threshold || !is.numeric(x))&&
				is.null(breaks) && !forceContinous)){
		
		if (!is.null(mapping))
			x<-mapuj(x, mapping)	
		
		discret<-buckety_stat(x, y, total=TRUE);
		
		#Sprawdzam, czy nie ma za dużo zmiennnych kategorycznych. Parametr określony z parametru.
		if(nrow(discret)-1 > no_stats_threshold)
			return("Too many categorical levels")
		
		#jeśli tylko jedna wartość, to zwracam komunikat
		if(nrow(discret) == 2)
			return("One value")
	
		## uzupełniam statystyki ##
		
		# ciągłe
		discret$od<-NA;
		discret$do<-NA;
		discret$srodek<-NA;
		
		#dyskretne
		nam<-rownames(discret)
		if (is.numeric(x)){
			nam[length(nam)]<-NA
			discret$discret<-as.numeric(nam)
		}
		else{
			nam[length(nam)]<-"<TOTAL>";
			discret$discret<-nam;
		}
		
		#mapowanie
		if(!is.null(mapping)) 
			discret$mapping_war<-mapping$war
		else
			discret$mapping_war<-NA
		
		discret<-discret[,c('nr','label','discret', 'mapping_war', 'od','srodek','do','n_good','pct_good','n_bad','pct_bad','n_obs','pct_obs',
						'br','woe','logit')]
		discret$predicted<-discret$br
	}
	## jeśli jest to zmienna ciągła
	else{
		discret<-numeric_var_treatment(x,y, special_val=special_val, NA_substit = NA_substit,
		    discrete_threshold=discrete_threshold,
				max_gleb=max_gleb,min_bucket=min_bucket,breaks=breaks,
				interactive=interactive, locfit=locfit, span=span, ...);
	}
	
	discret$label<-rownames(discret);
	return(discret);
}



#' Nie wiem co to robi
#' 
#' @param dane dane
#' @param mapowanie mapowanie 
#' @param czas czas
#' @param ...  dodatkowe parametry graficzne.
#' @return 
#' 
#' @author Michał Danaj
#' @export
univariate_anal_stats4<-function(dane, mapowanie, czas=lastDay(dane$reportingdate, unit = "quater"),...){
	dyskretne<-mapuj(dane,mapowanie)
	wynik<-univariate_anal_stats(dyskretne, dane$def, czas,...)
	
	#funkcja univariate_anal_stats zmienia kolejność, dlatego wracamy ją
	kolejnosc<-match(mapowanie$label,wynik$dyskretyzacja$label)
	
	#dodaję totala
	kolejnosc<-c(kolejnosc,length(kolejnosc)+1)
	wynik$dyskretyzacja<-wynik$dyskretyzacja[kolejnosc,]
	wynik$rozklady$obs_all_tbl<-wynik$rozklady$obs_all_tbl[kolejnosc,]
	wynik$rozklady$pct_all_tbl<-wynik$rozklady$pct_all_tbl[kolejnosc,]
	wynik$rozklady$avg_t_tbl<-wynik$rozklady$avg_t_tbl[kolejnosc,]
	
	wynik$dyskretyzacja<-cbind(wynik$dyskretyzacja[,1:3], 
			mapping_war=c(as.character(mapowanie$war),""),
			wynik$dyskretyzacja[,4:length(wynik$dyskretyzacja)])
	wynik$dyskretyzacja$discret<-""
	wynik$dyskretyzacja$discret[nrow(wynik$dyskretyzacja)]="<TOTAL>"
	wynik
}





#' Grupuje zmienne i generuje dla nich listę statystyk
#' 
#' Funkcja grupuje zmienne ciągłe, zgodnie z domyślnymi parametrami. Na grupach oraz na zmiennych kategorycznych 
#' wylicza statytystyki i przedstawia je w formie listy, osobno dla każdej ze zmiennych. Wykorzystuje do tego funkcję
#' \code{\link{univariate_anal_stats}}.
#' @param x_df \code{data.frame} ze zmiennymi do anliz.
#' @param y \code{vector} z wartościami zmiennej odpowiedzi.
#' @param vsub_bool \code{vector} logiczny określający, które zmienne z \code{df} powinny zostać zanalizowane.
#' @param vsub_list \code{vector} z nazwami zmiennych z \code{df}, które powinny zostać zanalizowane.
#' @param vsub_rola \code{vector} z rolą każdej ze zmiennych. Zostaną zanalizowane zmienne, dla których rola 
#' 	ustawiona jest na \code{'explanatory'}. 
#' @param proby \code{data.frame} z wektorami logicznymi, określającymi podpróby na których należy przeanalizować podpróby.
#' @param czas \code{vector} ze zmienną czasową, po której zostaną podzielone analizy.
#' @param min_bucket minimalna liczba obserwacji w buckecie, przy dzieleniu drzewem.
#' @param ... dodatkowe parametry do funkcji \code{\link{univariate_anal_stats}}. 
#' @return lista ze statstykami dla każdej ze zmiennych określonych przez któryś z wektorów vsub\*. Nazwy elementów listy
#' 	są nazwami zmiennych.
#' @seealso \code{\link{univariate_anal_stats}}
#' 
#' @author Michal Danaj
#' @export
univariate_loop<-function(x_df
		, y
		, vsub_bool = rep(TRUE, ncol(x_df))
		, vsub_list = NULL
		, vsub_rola = NULL
		, proby = rep(TRUE, length(y))
		, czas = rep(1, length(y))
		, min_bucket=floor(0.05*length(y))
		, ...){
	
	zmienne_names <- names(x_df)
	
	#które zmienne do analizy
	if (!is.null(vsub_rola))
		vsub_bool <- vsub_rola=='explanatory'
	
	if (!is.null(vsub_list))
		vsub_bool <- names(x_df) %in% vsub_list
	
	wyniki<-list()
	
	for (zmienna in zmienne_names)
	{
		
		i<-which(zmienne_names %in% zmienna)
		if (vsub_bool[i]==FALSE)
			next;
		
		
		x<-x_df[,i]
		
		print ('                                                                              ')
		print ('------------------------------------------------------------------------------')
		print ('                                                                              ')
		print(paste(i,"/",length(zmienne_names),": ",zmienna))
		
		
		wyniki[[zmienna]]<-univariate_anal_stats(x, y,
				proby=proby,
				czas=czas,
				min_bucket=min_bucket,
				...
		)
		
	}
	
	wyniki
}	


