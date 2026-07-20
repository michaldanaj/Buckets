# TODO: Add comment
# 
# Author: Michał Danaj
###############################################################################



#' Generuje raport
#' 
#' Generuje raport
#' @param wyniki lista z wynikami dyskretyzacji itp, z funkcji \code{\link{univariate_anal_stats}}
#' @param dir katalog z raportem, jako pełna bezwzględna ścieżka! Katalog musi być stworzony. Domyślnie \code{\link{tempdir()}}. 
#' @param kolejnosc kolejność wg której zmienne mają być wyświetlone.
#' @param show czy na koniec wyświetlić raport w przeglądarce. Domyślnie \code{TRUE}. 
#' @param scale Skala osi OY.   
#' 
#' @author Michał Danaj
#' @export
genRaport<-function(wyniki, dir=tempdir(), kolejnosc=1:length(wyniki), show=TRUE, scale=c(0,0.2)){
	
	makeCSSFile(dir)
	
	R2HTML::HTMLStart(dir , "univariate", HTMLframe=TRUE, Title="Univariate analysis",
			echo=TRUE);
	R2HTML::HTMLStop();
	
	
	plik_main<-R2HTML::HTMLInitFile(dir, 'univariate_main', CSSFile='R2HTML MD.css');
	 
	
	genRaportBody(wyniki, kolejnosc, dir, plik_main, scale)
	genRaportMenu(wyniki, dir)
	
	if(show)
		browseURL(paste(dir, 'univariate.html', sep='/'))
}


genRaportBody<-function(wyniki, kolejnosc, dir, plik_main, scale){
	
	for (i in 1:length(wyniki)){
		#for (i in 1:1){
		wynik<-wyniki[[kolejnosc[i]]];
		nazwa_zmiennej<-names(wyniki)[kolejnosc[i]];
		
		cat(sprintf('<a name="%s">', nazwa_zmiennej), file=plik_main, append=TRUE)
		R2HTML::HTML.title(nazwa_zmiennej);
		cat('</a>', file=plik_main, append=TRUE)
		#    windows();
		
		#jeśli z jakiegoś powodu zmienna nie została wygenerowana;
		if (typeof(wynik$dyskretyzacja)=="character"){
			R2HTML::HTML(wynik$dyskretyzacja);
			next
		}
			
		
		### dyskryminacja ###
		R2HTML::HTML.title("Discrimination GINI", HR=3);
		if (!is.null(wynik$rozklady$pct_all_tbl)){
			do_wykresu<-reshape::melt(wynik$dyskryminacja)
			do_wykresu<-do_wykresu[do_wykresu$X2!='AR_calosc',];
			X1_order<-ordered(do_wykresu$X1, levels=rownames(wynik$dyskryminacja));
			print(str(do_wykresu));
			print(lattice::xyplot(value ~ X2 , group = X1_order, data=do_wykresu, type='b',
							xlab="Date", ylab="GINI", main=nazwa_zmiennej));
			print('1');
			R2HTML::HTMLplot(Caption = "", file = plik_main, append = TRUE, GraphDirectory = dir,   GraphFileName = paste(nazwa_zmiennej, ' discrimination'), GraphSaveAs = "png", GraphBorder = 1,  Align = "center",
					Width = 400, Height = 400, WidthHTML = NULL,     HeightHTML = NULL, GraphPointSize = 12, GraphBackGround = "white",     GraphRes = 72)
			
			R2HTML::HTML(wynik$dyskryminacja);
		}
		
		###   rysunek PIT/TTC   ###
		R2HTML::HTML.title("Point in Time or Through the Cycle", HR=3);
		if (!is.null(wynik$rozklady$avg_t_tbl)){
			plot(wynik$rozklady$avg_t_tbl['TOTAL',-ncol(wynik$rozklady$avg_t_tbl)], main="PIT/TTC",
					ylab="Average target", xlab="Date");
			points(wynik$rozklady$estim, col="green")
			R2HTML::HTMLplot(Caption = "Does changes in variable distribution follow changes of target over time?",
					file = plik_main, append = TRUE, GraphDirectory = dir,   GraphFileName = paste(nazwa_zmiennej, 'cycle'), GraphSaveAs = "png", GraphBorder = 1,  Align = "center",
					Width = 400, Height = 400, WidthHTML = NULL,     HeightHTML = NULL, GraphPointSize = 12, GraphBackGround = "white",     GraphRes = 72)
			
			R2HTML::HTML(    t(data.frame("Observed target" = wynik$rozklady$avg_t_tbl['TOTAL',-ncol(wynik$rozklady$avg_t_tbl)],
									"Estimated target" = wynik$rozklady$estim))
			);
		}
		
		###   dyskretyzacja   ###
		R2HTML::HTML.title("Buckets", HR=3);
#    dev.off();
		windows(1400,700);
		par(mfrow=c(1,2));
		
		#nie wiem, czemu był tu wymóg rysowania tylko ciągłych wartości
		#zobaczymy, jak to będzie po usunięciu tego.
		#ciagle<-nchar(wynik$dyskretyzacja$discret)==0
		#drzewo_plot(wynik$dyskretyzacja[ciagle,], xlab=nazwa_zmiennej, ylab="Mean LGD",
		#		main=paste(nazwa_zmiennej,"discretization"));
		drzewo_plot(wynik$dyskretyzacja, xlab=nazwa_zmiennej, ylab="Average target",
				main=paste(nazwa_zmiennej,"discretization"));
		ile_row<-nrow(wynik$dyskretyzacja);
		b<-barplot(wynik$dyskretyzacja$pct_obs[-ile_row], names.arg=rownames(wynik$dyskretyzacja)[-ile_row], xlab=nazwa_zmiennej,
				ylab='Distribution',main="Distribution with target");
		par(usr=c(par()$usr[1:2], scale))
		lines(b, wynik$dyskretyzacja$br[-ile_row],type="o", col="red", lty="solid", pch="x")
		axis(4)
		R2HTML::HTMLplot(Caption = "Results of discretization", file = plik_main, append = TRUE, GraphDirectory = dir,   GraphFileName = paste(nazwa_zmiennej, 'tree'), GraphSaveAs = "png", GraphBorder = 1,  Align = "center",
				Width = 800, Height = 400, WidthHTML = NULL,     HeightHTML = NULL, GraphPointSize = 12, GraphBackGround = "white",     GraphRes = 72)
		
		par(mfrow=c(1,1));
		R2HTML::HTML(wynik$dyskretyzacja);
		
		###   rozkłady    ###
		R2HTML::HTML.title("Distribution of buckets", HR=3);
		if (!is.null(wynik$rozklady$pct_all_tbl)){
			do_wykresu<-reshape::melt(wynik$rozklady$pct_all_tbl)
			do_wykresu<-do_wykresu[do_wykresu$X1!='TOTAL' & do_wykresu$X2!='TOTAL',];
			X1_order<-ordered(do_wykresu$X1, levels=rownames(wynik$rozklady$pct_all_tbl));
			
			#dev.off();
			#png(filename = paste(dir,"xxx.png",sep="/"), width = 480, height = 480)
			print(lattice::barchart(value ~ X2|X1_order , stack=TRUE, data=do_wykresu, main=paste("Distribution of", nazwa_zmiennej),
							xlab='Date', ylab='Percent in given date'))
			#plot(1:10);
			#dev.off()
			
			#
			R2HTML::HTMLplot(Caption = "", file = plik_main, append = TRUE, GraphDirectory = dir,   GraphFileName = paste(nazwa_zmiennej, 'distribution'),
					GraphSaveAs = "png", GraphBorder = 1,  Align = "center",
					Width = 800, Height = 400, WidthHTML = NULL,     HeightHTML = NULL, GraphPointSize = 12, GraphBackGround = "white",     GraphRes = 72)
			
			R2HTML::HTML(wynik$rozklady$obs_all_tbl, caption="Number of observations");
			R2HTML::HTML(wynik$rozklady$pct_all_tbl, caption="% share at given date");
			
			#     średni target    #
			R2HTML::HTML.title("Average target", HR=3);
			do_wykresu<-reshape::melt(wynik$rozklady$avg_t_tbl)
			do_wykresu<-do_wykresu[do_wykresu$X1!='TOTAL' & do_wykresu$X2!='TOTAL',];
			X1_order<-ordered(do_wykresu$X1, levels=rownames(wynik$rozklady$pct_all_tbl));
			print(lattice::xyplot(value ~ X2|X1_order , data=do_wykresu, type='b', xlab="Date", ylab="Average target", main=nazwa_zmiennej,
							strip=lattice::strip.custom(bg='green')));
			
			R2HTML::HTMLplot(Caption = "", file = plik_main, append = TRUE, GraphDirectory = dir,   GraphFileName = paste(nazwa_zmiennej, 'target by bucket'), GraphSaveAs = "png", GraphBorder = 1,  Align = "center",
					Width = 800, Height = 400, WidthHTML = NULL,     HeightHTML = NULL, GraphPointSize = 12, GraphBackGround = "white",     GraphRes = 72)
			print(lattice::xyplot(value ~ X1_order |X2 , data=do_wykresu, type='b', xlab="Bucket", ylab="target", main=nazwa_zmiennej));
			R2HTML::HTMLplot(Caption = "", file = plik_main, append = TRUE, GraphDirectory = dir,   GraphFileName = paste(nazwa_zmiennej, 'target over time'), GraphSaveAs = "png", GraphBorder = 1,  Align = "center",
					Width = 800, Height = 400, WidthHTML = NULL,     HeightHTML = NULL, GraphPointSize = 12, GraphBackGround = "white",     GraphRes = 72)
			
			R2HTML::HTML(wynik$rozklady$avg_t_tbl, caption="Average target");
			
			dev.off();
		}
	}
}



###########   generuję plik z menu   #################

genRaportMenu<-function(wyniki, dir){
	
	plik_menu<-R2HTML::HTMLInitFile(dir, 'univariate_menu');
	
	GINI<-sapply(wyniki, function(wynik){
				gini<-wynik$dyskryminacja[1,'AR_calosc']
				if (is.null(gini))
					return(NA);
				return(gini);
			})
	
	
	#GINI
	kolej<-rev(order(GINI));
	R2HTML::HTML.title('Sortowanie po GINI');
	for (i in 1:length(wyniki)){
		nazwa_zmiennej<-names(wyniki)[kolej[i]];
		cat(sprintf('<a href="univariate_main.html#%s" target=main>%s (%f)</a></br>\n',nazwa_zmiennej,nazwa_zmiennej, round(GINI[kolej[i]],3))
				, file=plik_menu, append=TRUE);
	}
	
	
	
	
	#alfabetycznie
	kolejnosc<-order(names(wyniki));
	R2HTML::HTML.title('Sortowanie Alfabetyczne');
	for (i in 1:length(wyniki)){
		nazwa_zmiennej<-names(wyniki)[kolejnosc[i]];
		cat(sprintf('<a href="univariate_main.html#%s" target=main>%s</a></br>\n',nazwa_zmiennej,nazwa_zmiennej)
				, file=plik_menu, append=TRUE);
	}
	
	R2HTML::HTMLEndFile(plik_menu)
}


###########   CSS   #################


makeCSSFile<-function(dir){
	tekst<-'
			body {
			background: #FFFFFF;
			color: #000000;
			font-family: Verdana, Arial, Helvetica, sans-serif;
			font-size: 10pt;
			font-weight: normal
			}
			
			.tablesort {
			cursor: pointer;
			behavior: url(tablesort.htc);
			-moz-binding: url(tablesort.htc);
			}
			
			H1 {
			font-family: Arial, Helvetica, sans-serif;
			font-size: 30pt;
			font-style: normal;
			font-weight: bold;
			color: #3333CC;
			background: #004080;
			text-align: center;
			margin: 10pt 2.5%
			}
			
			H2 {
			font-family: Arial, Helvetica, sans-serif;
			font-size: 17pt;
			font-style: normal;
			font-weight: bold;
			color: #FFFFFF;
			background: #0050d0;
			text-align: center
			}
			
			H2.index {
			font-family: Arial, Helvetica, sans-serif;
			font-size: 17pt;
			font-style: normal;
			font-weight: normal;
			color: #FFFFFF;
			background: #0050d0;
			text-align: center;
			margin: 10pt 5%
			}
			
			H3 {
			font-family: Arial, Helvetica, sans-serif;
			font-size: 14pt;
			font-style: normal;
			font-weight: bold;
			text-align: center;
			color: #004080
			}
			
			H4 {
			font-family: T, Helvetica, sans-serif;
			font-size: 10pt;
			font-style: normal;
			font-weight: bold;
			color: #000000;
			line-height: 16pt
			}
			
			LI {
			font-family: Verdana, Arial, Helvetica, sans-serif;
			font-size: 10pt
			}
			
			A {
			font-family: Verdana, Arial, Helvetica, sans-serif;
			font-size: 10pt;
			text-decoration: none
			}
			
			.caption {
			font-style: italic
			}
			
			.title2 {
			font-family: Arial, Helvetica, sans-serif;
			font-size: 14pt;
			font-style: normal;
			font-weight: bold;
			color: #004080
			}
			
			.equation{
			font-weight: bold;
			}
			
			.command {
			font-family=verdana, arial;
			color=red
			}
			
			.partitle {
			font-family=verdana, arial;
			font-weight: bold
			}
			
			XMP {
			font-family: Verdana, Arial, Helvetica, sans-serif;
			font-size: 10pt
			}
			
			.function {
			font-family=courier;
			color=blue;
			font-size: 10pt
			}
			
			TR {
			font-family: Arial, Helvetica, Times, Helvetica, sans-serif;
			font-size: 10pt;
			font-style: normal;
			padding: 0 0
			}
			
			TR.firstline {
			color: #FFFFFF;
			background: #000000;
			text-align=center;
			font-weight: bold
			}
			TR.ListBackTitle {
			color: #FFFFFF;
			background: #000000;
			text-align=left;
			font-weight: bold
			}
			TD {
			background=#FFFFFF;
			padding: 0 0
			}
			TD.ListBackMain {
			background: #E0E0E0;
			padding: 0 0
			}
			TD.firstcolumn {
			padding: 5 10;
			background: #C0C0C0;
			text-align=right
			}
			TD.cellinside {
			padding: 5 10;
			background: #FFFFFF;
			text-align=right
			}
			/* CORRELATION MATRIX TRAFFIC HIGHLIGHT*/
			TD.corvarname {
			background-color="#FFFFFF";
			color=black;
			height: 1.1cm;
			text-align: right;
			font-weight: bold
			}
			TD.corsep {
			width: 0.5cm
			}
			TD.cordiag {
			background-color=#fffff;
			color=white
			}
			TD.cor0 {
			background-color="#FFFFFF";
			color=black;
			width: 1.1cm;
			height: 1.1cm;
			text-align: center
			}
			TD.cor1 {
			background-color="#E6E6E6";
			color=black;
			width: 1.1cm;
			height: 1.1cm;
			text-align: center
			}
			TD.cor2 {
			background-color="#CCCCCC";
			color=black;
			width: 1.1cm;
			height: 1.1cm;
			text-align: center
			}
			TD.cor3 {
			background-color="#B3B3B3";
			color=black;
			width: 1.1cm;
			height: 1.1cm;
			text-align: center
			}
			TD.cor4 {
			background-color="#999999";
			color=black;
			width: 1.1cm;
			height: 1.1cm;
			text-align: center
			}
			TD.cor5 {
			background-color="#808080";
			color=white;
			width: 1.1cm;
			height: 1.1cm;
			text-align: center
			}
			TD.cor6 {
			background-color="#666666";
			color=white;
			width: 1.1cm;
			height: 1.1cm;
			text-align: center
			}
			TD.cor7 {
			background-color="#4D4D4D";
			color=white;
			width: 1.1cm;
			height: 1.1cm;
			text-align: center
			}
			TD.cor8 {
			background-color="#333333";
			color=white;
			width: 1.1cm;
			height: 1.1cm;
			text-align: center
			}
			TD.cor9 {
			background-color="#1A1A1A";
			color=yellow;
			width: 1.1cm;
			height: 1.1cm;
			text-align: center
			}
			TD.cor10 {
			background-color="#000000";
			color=yellow;
			width: 1.1cm;
			height: 1.1cm;
			text-align: center
			}'
	
	nazwa<-paste(dir,'/R2HTML MD.css', sep='')
	file.create(nazwa)
	cat(tekst, file=nazwa)
}


