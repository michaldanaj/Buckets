# TODO: Add comment
# 
# Author: Michał Danaj
###############################################################################




#' Rysuje kalibrację modelu
#' 
#' Rysuje kalibrację modelu
#' @param score Zmienna na osi OX.
#' @param response Zmienna celu.
#' @param estim Wyestymowane wartość przez model(e). Może byś wektor lub lista wektorów.
#' @param plt_type Co ma być na osi OY. Czy średnia z wartości \code{response}, czy logit.
#' @param ylab Opis osi OY.
#' @param xlab Opis osi OX.
#' @param ... Parametry graficzne.
#' @return 
#' 
#' @author Michał Danaj
#' @export
plotCalibr<-
		function (score, response, estim, plt_type = c("br", "logit"), ylab=plt_type,
				xlab="score",    ...)
{
	plt_type <- match.arg(plt_type)
	if (!is.list(estim))
		estim <- list(estim)

	buck <- reg_nieparam(score, response, plt_type = plt_type, ylab = ylab,
			xlab = xlab, ...)
	grupy_skala <- data.frame(od = buck$od, do = buck$do, fitted = buck$nr)
	grupy <- przypisz(score, grupy_skala)
	kolejnosc <- order(score)
	
	for (i in 1:length(estim)) {
		estim_grp <- tapply(estim[[i]], grupy, mean)
		
		buck <- cbind(buck, estim_grp)
		names(buck) <- c(names(buck)[1:(ncol(buck) - 1)], paste(names(estim[i]),
						"br", sep = "_"))
		if (plt_type == "logit") {
			estim[[i]] <- logit(estim[[i]])
			buck <- cbind(buck, logit(estim_grp))
			names(buck) <- c(names(buck)[1:(ncol(buck) - 1)],
					paste(names(estim[i]), "logit", sep = "_"))
		}
		lines(score[kolejnosc], estim[[i]][kolejnosc], col = i +
						1, lty = i + 1, pch = i + 19)
	}
	legend(x = "topright", col = 1:(length(estim) + 1), lty = 1:(length(estim) +
						1), legend = c("Dependence from data", names(estim)),
			bty = "n", inset = 0.05)
	return(buck)
}




#' Generuje tabelę z coarse classing do dokumentacji
#' 
#' Jeśli Woe lub IV wychochą +-Inf, to wstawia zamiast tego NA
#' @param wyniki lista z wynikami dyskretyzacji
#' @export 
makeCoarseClassingTables<-function(wyniki){
	lapply(wyniki, function(z){
				x<-z$dyskretyzacja;
				last_row<-nrow(x);
				x$GB_ODDS<-x$n_good/x$n_bad;
				gb_odds_total<-x$GB_ODDS[last_row];
				
				temp<-x$GB_ODDS/gb_odds_total;
				x$GB_index<-temp;
				
				print(x$GB_ODDS)
				x$GB_index[na.subst(x$GB_ODDS>gb_odds_total,FALSE)]<-paste(round(temp[na.subst(x$GB_ODDS>gb_odds_total,FALSE)]*100), 'G', sep='')
				x$GB_index[na.subst(x$GB_ODDS<=gb_odds_total, FALSE)]<-paste(round(temp[na.subst(x$GB_ODDS<=gb_odds_total, FALSE)]*100), 'B', sep='')
				
				x$IV<-(x$pct_good-x$pct_bad)*x$woe;
				
				#zamienia nieskończoności na braki danych
				x$IV[abs(x$IV)==Inf]<-NA;
				x$woe[abs(x$woe)==Inf]<-NA;
				
				x$IV[last_row]<-sum(x$IV[-last_row], na.rm=TRUE);
				x<-x[,c('label','n_obs','pct_obs','n_good','pct_good','n_bad','pct_bad','GB_ODDS','GB_index',
								'br','woe', 'IV')]
				x$discret[x$discret=='']<-x$label[x$discret==''];
				names(x)<-c('Coarse Classes','# Applicants','% Applicants',
						'# Good','% Good','# Bad','% Bad','GB Odds','GB Index',
						'LGD','WoE', 'IV')
				x<-unique(x);
				return(x);
			})
}






#' Przypisuje woe na podstawie br z bucketa.
#' 
#' Przypisuje woe na podstawie br z bucketa. Korzysta z funkcji \code{woe}, która 
#' w razie niewystąpienia w buckecie wartości z klasy 0 lub 1, przyjmuje wartość 0.5
#' 
#' @param bucket_list - lista z opisami dyskretyzacji. Nazwy elementów listy powinny być zgodne z nazwami zmiennych w \code{data}.
#' @param data - \code{data.frame} z oryginalnymi zmiennymi.
#' @param vars - wektor z nazwami zmiennych do ograniczenia.
#' @param varname_sufix - Sufix wynikowych zmiennych.
#' @export
przypisz_woe_z_listy<-function(bucket_list, data, vars=names(bucket_list), varname_sufix='woe'){
	
	data_out<-NULL
	
	for (zmienna in names(bucket_list)){
		
		if (!(zmienna %in% vars)) 
			next;
		
		####   wyliczam   woe    ######
		
		#Wyciągam element listy
		bucket<-bucket_list[[zmienna]]
		
		#jeśli badów lub goodów jest 0, to przyjmuję że jest 0.5	
		
		pct_good = (pmax(bucket$n_good,0.5))/(max(bucket['TOTAL','n_good'],0.5))
		pct_bad = (pmax(bucket$n_bad,0.5))/(max(bucket['TOTAL','n_bad'],0.5))
		woe = log(pct_good/pct_bad)
		
		
		####   przypisuję woe    ######
		
		woe<-przypisz2(data[,zmienna],
				bucket_list[[zmienna]], 
				fitted=woe,
				NA_subst = numeric_var_treatment.params$NA_substit,
				interpol=FALSE)
		
		if (is.null(data_out))	{
			data_out <- data.frame(woe)
			names(data_out)<-zmienna
		}
		else
			data_out[,zmienna] <- c(woe)
		
	}
	
	names(data_out)<-paste(names(data_out), varname_sufix, sep="_")
	data_out
}	


#' Na podstawie listy z bucketami przypisuje odpowiednie wartości
#' 
#' Na podsatwie listy \code{bucket_list} przypisuje zmiennych \code{vars} z tabeli \code{data}
#' wartości znajdujące się w kolumnie \code{fitted} bucketów. Wartości przypisuje do kolumn
#' o nazwie z połącznia nazw \code{vars} i \code{varname_sufix}.
#' @param bucket_list Lista bucketów.
#' @param data Dane.
#' @param vars Zmienne do przypisania.
#' @param colname Nazwa kolumny z bucketa, na podstawie którego należy dopisać wartości.
#' @param varname_sufix Sufiks do nazw nowych zmiennych.
#' @param sep_sufix Parametr umożliwiający zmianę separatora sufiksa. Domyślnie podkreślenie - została ustalona
#' 		w celu zachowania kompatybilności z wersją funkcji bez tego parametru.
#' @return 
#' 
#' @author Michał Danaj
#' @export
przypisz_z_listy<-function(bucket_list, data, vars=names(bucket_list), colname='fitted', varname_sufix=colname, sep_sufix="_"){
	
	data_out<-NULL
	
	for (zmienna in names(bucket_list)){
		
		if (!(zmienna %in% vars)) 
			next;
		
		
		#Wyciągam element listy
		bucket<-bucket_list[[zmienna]]
		#bucket<-bucket[rownames(bucket)!='TOTAL',]

		#Sprawdzam, czy nie jest to komunikat o niespełnieniu kryteriów dyskretyzacji. Jeśli tak, 
		#to wypisuję stosowny komunikat.
		if (class(bucket)=="character"){
			warning(paste("Zmienna", zmienna, "nie ma obiektu z dysretyzacją. Przyczyna braku dyksretyzacji to:", bucket,". Zmienna ta zostanie pominięta w liście z wynikiem zwróconym przez funkcję."))
			next;
		}
			
		#jeśli badów lub goodów jest 0, to przyjmuję że jest 0.5	
		
		fitted = bucket[,colname]
		
		####   przypisuję woe    ######
		
		fitted_x<-przypisz2(data[,zmienna],
				bucket_list[[zmienna]], 
				fitted=fitted,
				NA_subst = numeric_var_treatment.params$NA_substit,
				interpol=FALSE)
		
		if (is.null(data_out))	{
			data_out <- data.frame(fitted_x)
			names(data_out)<-zmienna
		}
		else
			data_out[,zmienna] <- c(fitted_x)
		
	}
	
	names(data_out)<-paste(names(data_out), varname_sufix, sep=sep_sufix)
	data_out
}	




#' Wyświetla w gridzie wartości korelacji zmiennych z modelu 
#' 
#' @param model dopasowany model. 
#' @param data  dane.
#' @return 
#' 
#' @author Michał Dnaaj
#' @export
korelacje_zmiennych<-function(model, data){
	zmienne<-names(coef(model))[-1]
	edit(cor(data[,zmienne]))
}

