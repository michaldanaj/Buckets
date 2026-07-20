# TODO: Add comment
# 
# Author: Michał Danaj
###############################################################################



#' Odchylenie modelu
#' 
#' @param predicted prawdopodobieństwo.
#' @param target target.
#' @return 
#' 
#' @author Michał Danaj
#' @export
model_dev<-function(predicted, target){
	-2*sum(target*log(predicted)+(1-target)*log(1-predicted))
}


#' Zastępuje braki danych
#' 
#' @param x x
#' @param val jaką wartością zastąpić brak danych. 
#' @return 
#' 
#' @author Michał Danaj
#' @export
na.subst<-function(x, val){
	x[is.na(x)]<-val;
	x
}


#' Wrzuca dane do schowka, do wklejenia do Excela
#' @param table \code{data.frame} do przekazania do Excela. 
#' @param row.names  czy przekazać również wiersze.
#' @return 
#' 
#' @author Michał Danaj
#' @export
to_excel<-function(table, row.names=FALSE){
	write.table(table, file='clipboard', sep='\t', row.names=row.names, dec=',')
}


#do skasowania!!! 
#' Liczy WoE
#' 
#' @param br wartość prawdopodobieństwa.
#' @return 
#' 
#' @author Michał Danaj
#' @export
woe<-function(br){
	br[br==0]<-0.5/length(br)
	br[br==1]<-(length(br)-0.5)/length(br)
	log(br/(1-br))
}


#' Zwraca ostatni dzień podanego okresu.
#'
#' @param dat Obiekt klasy \link{Date}, lub dający się skonwertować funkcją \link{as.Date}.
#' @param unit Jednostka czasu, która ma być uwzględniona przy wyznaczaniu ostatniego dnia.
#'  Możliwe wartości: \code{c("month","quater", "year")}, które odpowiadają ostatniemu dniu miesiąca, kwartału, roku,
#'  do którego należy data \code{dat}.
#' @param ... Opcje do funkcji \link{as.Date}.
#' @author Michał Danaj
#' @export
lastDay<-function(dat, unit=c("month","quarter", "half_year", "year"),...) {
	
	# Wstępne przygotowania
	unit<-match.arg(unit);
	
	dat<-as.Date(dat,...);
	
	# Pobieram rok i miesiąc
	rok<-as.numeric(format(dat, '%Y'));
	mies<-as.numeric(format(dat, '%m'));
	
	# Jeśli ostatni dzień kwartału, to wybieram ostatni miesiąc kwartału
	if (unit=="quarter")
		mies<-((mies-1)%/%3+1)*3
	
	if (unit=="half_year")
		mies<-((mies-1)%/%6+1)*6
	
	if (unit=="year")
		mies<-12;
	
	# Właściwy algorytm
	ostatni<-mies==12;
	mies[!ostatni]<-mies[!ostatni]+1;
	mies[ostatni]<-1;
	rok[ostatni]<-  rok[ostatni]+1;
	
	# zamiana na datę
	new_date<-as.Date(ISOdate(rok, mies, 1));
	return(new_date-1);
}


#' Na podstawie nazwy zmiennej celu i zmiennych objaśniających, tworzy formułę modelu
#' 
#' @param target zmienna celu 
#' @param vars zmienne objaśniające
#' @return forumła do zastosowania w modelu
#' 
#' @author Michał Danaj
#' @export
make_model_formula<-function(target, vars){
	
	#jeśli w nazwach zmiennych jest nazwa targetu, to ją usuwam
	czy_jest_target<-target==vars
	vars<-vars[!czy_jest_target]
	
	#suma zmiennych
	suma_zmiennych<-paste(vars, collapse = ' + ')
	
	#wynik
	formula(paste(target, suma_zmiennych, sep='~'))
}
