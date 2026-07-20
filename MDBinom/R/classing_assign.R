# Funkcje zaczynające się na univariate
# 
# Author: Piotr
###############################################################################


# TODO  BUG! Jeśli bucket wiersze nie będą posortowane, to będą błędne wyniki, bo później
# sortujemy i odnosimy się tym porządkiem do początkowego porządku
# TODO obsługa jednego przedziału dla interpolacji
# TODO do przetestowania zaokrąglenia
#' Przypisuje wartości zdefiowane w \code{bucket} 
#' 
#' Znajduje przedział opisany w \code{bucket}, do którego należy \code{x} i 
#' przypisuje wartość z \code{bucket_fitted}. Ewentualnie wykonuje interpolację.
#'
#' Jeśli \code{interpor=TRUE}, wykonuje interpolację, której węzłami są 
#' środki przedziałów, pomiędzy którymi leży \code{x}, a wartościami
#' \code{bucket$fitted} odpowiadające tym środkom.  
#' 
#' Aby zapobiec skutkom błedów numerycznych, domyślnie wartości \code{x}, jak i \code{bucket$srodek},
#' \code{bucket$od}, \code{bucket$do} zaokrąglone są do 15-tej cyfry znaczącej.
#' 
#' @param x Wektor zmiennych numerycznych. 
#' @param bucket \code{data.frame} z kolumnami:
#'		\itemize{
#' 			\item od. Dolny kraniec przedziału.
#' 			\item do Górny kreniec przedziału.
#' 			\item srodek Środek przedziału.
#' 			\item fitted Wartość przypisana do danego przedziału.
#' 		}
#' @param interpol Czy przeprowadzić interpolację?
#' @param round Czy zaokrąglać. Domyślnie tak.
#' @param digits Do której cyfry znaczącej zaokrąglać. Domyślnie do 15-tej.
#' 
#' @return  Wektor wartości.
#' @seealso \code{\link{przypisz}}
#' @seealso \code{\link{mapuj}}
#' @author Michał Danaj
#' 
#' @examples
#' 		n<-1000;
#' 		x1<-rnorm(n);
#' 		y<-(x1+rnorm(n))<0;
#' 		
#' 		b<-buckety_br(x1,y,10);
#' 		b$fitted<-b$br;
#' 		
#' 		y_fitted<-przypisz(x1, b, interpol=FALSE)
#' 		plot(x1, y_fitted)
#' 		
#' 		y_fitted<-przypisz(x1, b, interpol=TRUE)
#' 		plot(x1, y_fitted) 
#' @seealso \code{\link{przypisz2}}
#' 
#' @export

przypisz<-function (x, bucket, interpol = FALSE, round=TRUE, digits=15)
{
	#zaokrąglam
	if (round){
		x <- signif(x,digits)
		bucket$srodek <- signif(bucket$srodek,digits)
		bucket$od <- signif(bucket$od,digits)
		bucket$do <- signif(bucket$do,digits)
	}
	
	n <- dim(bucket)[1]
	{
		if (interpol) {
			x[x < min(bucket$srodek)] <- min(bucket$srodek)
			x[x > max(bucket$srodek)] <- max(bucket$srodek)
			przedzial <- findInterval(x, bucket$srodek, rightmost.closed = TRUE,
					all.inside = TRUE)
			od <- bucket$srodek[przedzial]
			do <- bucket$srodek[przedzial + 1]
			wynik <- (x - od)/(do - od) * bucket$fitted[przedzial +
							1] + (do - x)/(do - od) * bucket$fitted[przedzial]
		}
		else {
			granice = sort(unique(c(bucket$od, bucket$do)))
			
			if (length(granice)==1)
				granice<-c(granice,granice);
			przedzial <- findInterval(x, granice, rightmost.closed = TRUE,
					all.inside = TRUE)
			wynik <- bucket$fitted[przedzial]
		}
	}
	wynik
}


# TODO  BUG! Jeśli bucket wiersze nie będą posortowane, to będą błędne wyniki,
# TODO Coś nie działa obsługa,  gdy w x są NA, np. zmienna Wiek
#' Przypisuje dyskretyzację
#' 
#' Przypisuje zmienne dyskretne oraz ciągłe w oparciu o podane przedziały, uwzględnia wartości
#' specjalne.
#' 
#' #' Aby zapobiec skutkom błedów numerycznych, domyślnie wartości \code{x}, jak i \code{bucket$srodek},
#' \code{bucket$od}, \code{bucket$do} zaokrąglone są do 15-tej cyfry znaczącej. Podobnie przy zmiennych
#' dyskretnych numerycznych. Więcej szczegółów w \code{\link{przypisz}}.
#' 
#'
#' @param x Wektor zmiennych numerycznych. 
#' @param bucket \code{data.frame} z bucketami.
#' @param interpol Czy przeprowadzić interpolację?
#' @param fitted Nadpisuje wektor \code{bucket$fitted} i na tej podstawie przypisuje wartości.  
#' @param NA_substit wartość do podstawienia za brak danych
#' @param round Czy zaokrąglać. Domyślnie tak.
#' @param digits Do której cyfry znaczącej zaokrąglać. Domyślnie do 15-tej.
#' @seealso \code{\link{przypisz}}
#' @seealso \code{\link{mapuj}}
#' @return  Wektor wartości.
#' @author Michał Danaj
#' @export
przypisz2<-function(x, bucket, interpol=FALSE, fitted=NULL, NA_substit=-2147483647, round=TRUE, digits=15)
{
	
	if (!is.null(fitted))
		bucket$fitted<-as.vector(fitted);
	
	if (is.null(bucket$fitted) & is.null(fitted))
		stop("Brak kolumny 'bucket$fitted' oraz parametru 'fitted'. Przynajmniej jeden powinien być uszupełniony.")
	
	if (is.factor(bucket$fitted))
		warning("przypisz2: Uwaga! bucket$fitted jest typu factor, co prowadzi do dziwnych wyników!");
	
	# Zokrąglenia.
	if (round & is.numeric(x)){
		x <- signif(x,digits)
		bucket$srodek <- signif(bucket$srodek,digits)
		bucket$od <- signif(bucket$od,digits)
		bucket$do <- signif(bucket$do,digits)	
		
		if (is.numeric(bucket$discret))
			bucket$discret<-signif(bucket$discret,digits)
	}	
	
	#inicjuję wektor z wynikami
	if (is.numeric(bucket$fitted))
		wynik<-rep(NA, length(x))
	else
		wynik<-rep("<NA>", length(x));
	#print(bucket)
	#Jeśli buckety określone są przez warunki mapujące
	jest_mapowanie=FALSE
	if (!is.null(bucket$mapping_war))
		if(any(!is.na(bucket$mapping_war)))
			jest_mapowanie=TRUE
	if (jest_mapowanie){
		mmm<-data.frame(war=bucket$mapping_war, label=bucket$fitted, discret=bucket$discret)
		mmm<-mmm[mmm$discret!="<TOTAL>",]
		wynik<-mapuj(x, mmm[,1:2])
	}
	else{
		#rozdzielam na wartości dyskretne i ciągłe
		ciagle_buck<-bucket[!is.na(bucket$od),];
		#wybieram dyskretne wartości i usuwam totala
		dyskretne_buck<-bucket[is.na(bucket$od) & bucket$discret!="<TOTAL>",];
		
		#czy są jakieś przedziały
		if (nrow(ciagle_buck)>0){
			if (interpol)
			{
				#obcinam zakresy do wartosci srodkow krancowych przedzialow
				x[x<min(ciagle_buck$srodek)]<-min(ciagle_buck$srodek);
				x[x>max(ciagle_buck$srodek)]<-max(ciagle_buck$srodek);
				
				przedzial<-findInterval(x, ciagle_buck$srodek, rightmost.closed = TRUE, all.inside = TRUE);
				
				od<-ciagle_buck$srodek[przedzial];
				do<-ciagle_buck$srodek[przedzial+1];
				wynik<-(x-od)/(do-od)*bucket$fitted[przedzial+1]+(do-x)/(do-od)*ciagle_buck$fitted[przedzial];
			}
			else
			{
				granice<-sort(unique(c(ciagle_buck$od, ciagle_buck$do)));
				if (length(granice)==1)
					granice<-c(granice, granice);
				przedzial<-findInterval(x, granice, rightmost.closed = TRUE, all.inside = TRUE);
				wynik<-ciagle_buck$fitted[przedzial];
			}
		}
		
		
		# Jeśli bucket zdefiniował obsługę braków danych, to wszystkie braki podmieniam
		# wartością specjalną oznaczajacą brak danych
		if (any(rownames(bucket)== NA_substit))
			x[is.na(x)]<-NA_substit;
		
		## I jeszcze wartości dyskretne lub specjalne
		# gdzie są
		spec_bool<- x %in% dyskretne_buck$discret;
		#jakie_sa
		spec_idx<- match(x, dyskretne_buck$discret);
		#nadpisuję
		wynik[spec_bool]<- dyskretne_buck$fitted[na.omit(spec_idx)];
		
	}
	
	
	#sprawdzam, czy są jakieś nieprzypisane wartości. Jeśli tak, to rzucam ostrzeżenie;
	if (is.numeric(bucket$fitted)){
		if (any(is.na(wynik)))
			warning("przypisz2: Nie wszystkie wartości zostały przypisane do bucketa. Pozostały braki danych.")	
	} 	else if(any(wynik=="<NA>"))
		warning("przypisz2: Nie wszystkie wartości zostały przypisane do bucketa. Pozostały braki danych,
						oznaczone jako <NA>'.")
	
	if (!is.numeric(wynik))
		wynik<-factor(wynik, levels=unique(bucket$fitted[bucket$discret!="<TOTAL>"]));
	
	wynik
}



#' Na podstawie zadanych warunków przypisuje etykiety
#' 
#' Na podstawie zadanych warunków w \code{mapping$war} przypisuje etykietę \code{mapping$label}.
#' W \code{mapping$label} nie może być wartości pustych stringów. Pusty string stosowany jest jako
#' brak danych. Jeśli dla jakieś elementu żaden warunek nie zachodzi, zwracany jest pusty string.    
#'
#' Możliwe jest określenie warunku \code{else} poprzez wpisanie w \code{mapping$war} stringa "else".
#'
#' @param data \code{data.frame} do którego będzie przypisywane mapowanie
#' @param mapping   \code{data.frame} z dwoma kolumnami znakowymi (lub do przerobienia przez \code{as.character})
#'			  \code{war} oraz \code{label}
#' @author Michał Danaj
#' @export
mapuj<-function(data, mapping){
	
	
	#jeśli nie znakowe, to przerabiam na znakowe
	if (!is.character((mapping$war)))
		mapping$war<-as.character(mapping$war)
	if (!is.character((mapping$label)))
		mapping$label<-as.character(mapping$label)	
	if (!is.data.frame(data))
		stop("Funkcja mapuj: Dane muszą być typu data.frame, z kolumnami wykorzystywanymi w warunkach mapujących!")
	
	wynik<-rep("", nrow(data))
	
	gdzie_else<-which(mapping$war=='else')
	
	#jeśli jest else, to go wydzielam
	if (length(gdzie_else>0)){
		lab_else<-mapping$label[gdzie_else]
		mapping<-mapping[-gdzie_else,]
	}
	
	text=sprintf("with(data,%s)",mapping$war)	
	for (i in 1:nrow(mapping)){	
		
		war<-eval(parse(text=text[i]))		
		wynik[war]<-mapping$label[i]
	}
	
	#jeśli jest else, to go stosuję
	if (length(gdzie_else)>0)
		wynik[wynik==""]<-lab_else
	
	#sprawdzam, czy coś się nie przypisało
	if (any(wynik==""))
		warning("mapuj: Nie wszystkie wartości zostały przypisane.")
	
	return(wynik)
}

