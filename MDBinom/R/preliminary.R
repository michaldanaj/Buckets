# TODO: Add comment
# 
# Author: Micha^3 Danaj
###############################################################################



#' Podstawowe statystyki faktorów
#' 
#' Wiele z faktorów zaczytanych bezpooerednio z pliku nie nadaje sie do modelowania. Funkcja umo?liwia wstepn^1 
#' analize faktorów. 
#' @param dane 
#' @return 
#' 
#' @author Micha^3 Danaj
#' @export
prel_factors<-function(dane){
	faktory<-sapply(dane, is.factor)
	faktory_data<-dane[,faktory]
	lapply(faktory_data, summary)
}



#' Wyoewietla zmienne numeryczne z wariancj^1 == 0
#' 
#' Wyoewietla zmienne numeryczne z jedn^1 wartooeci^1.
#' @param dane  
#' @return 
#' 
#' @author Micha^3 Danaj
#' @export
prel_numeric_sd0 <- function(dane){
	ciagle<-sapply(dane, function(x)is.numeric(x) & !is.factor(x))
	ciagle_names<-names(dane)[ciagle]
	
	sd_ciagle<-lapply(dane[,ciagle], sd, na.rm=TRUE)
	sd_ciagle<-nvl(sd_ciagle,0)
	
	do_wyrzucenia_ciagle<-names(dane)[ciagle][sd_ciagle==0]
	
	do_wyrzucenia_ciagle
}


#' Wyoewietla zmienne z wartooeci^1 o du?ym udziale w rozk^3adzie
#' 
#' Wyoewietla nazwy zmiennych, dla których istenieje taka wartooeae, ?e jej udzia^3 przekracza \code{rate}
#' próby. 
#' @param dane Dane.
#' @param rate Graniczny udzia^3 jedenej wartooeci.
#' @return 
#' 
#' @author Micha^3 Danaj
#' @export
prel_numeric_one_value <- function(dane, rate=0.98){
	
	ciagle<-sapply(dane, function(x)is.numeric(x) & !is.factor(x))
	ciagle_names<-names(dane)[ciagle]
	
	ciagle_table<-lapply(dane[,ciagle_names], function(x){
				z<-nvl(x,-9999999)
				max(table(z)/length(z))
			}
	)
	
	ciagle_table<-unlist(ciagle_table)	
	ciagle_names[ciagle_table> rate]
}



#' Wyoewietla wektory logiczne
#' 
#' Po wczytaniu danych z pliku tekstowego nie powinno byae wektorów logicznych w innych przypadkach ni?
#' pusty wektor (tego nie jestem tak na prawde teraz pewny). Funkcja wypisuje wszystkie wektory logiczne
#' ?eby mo?na by^3o je obejrzeae. 
#' @param dane 
#' @return 
#' 
#' @author Micha^3 Danaj
#' @export
prel_logical<-function(dane){
	names(dane)[unlist(sapply(dane,function(x)class(x)=='logical'))]
}


