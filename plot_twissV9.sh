#!/bin/sh
#               _(\   
#      _____   /  .|       ~ ~ ~ ~ ~
# >==.'TH FEL'   \_|  
#   /  |  |  | \/
#   |_ |  |  |__|
#  /  \|__|__/  \
#  \__/      \__/
#
# by Fahim Habib
###########################################
#
# plot_beta_emitt.sh
#
# Plots the (betac) and norm. emittance (en) functions in x
# and y plan along the beamline. Overlays the
# floormap of the beamline for reference.
#
# -----------------------------------------
#
# Usage Manuele
# =====
#
# ./plot_beta_emitt.sh base_name.slan [base_name.magn]
# ./plot_beta_emitt.sh base_name
#
# -----------------------------------------
#
# Arguments
# =========
#
# data.slan : Slice output file produced by
#            an Elegant simulation. The
#            file extension '.slan' may be
#            omitted. To produce Slice output file 
#			 you have to add set up &slice_analysis 
#			 in the main Elegant file %s.ele
#
# data.magn : Magnet output file produced by
#            an Elegant simulation. To produce
#			 Magnet output file you have to add
#			 magnets=%s.magn in to &run_setup 
#			 in the main Elegant file %s.ele	
#			 If the base name of this file is the
#            same as that of 'data.slan',
#            then this argument does not
#            need to be provided.
#
###########################################

# usage
if ! [ $1 ]; then
    echo "usage:"
    echo "plot_twiss.sh data.slan [data.mag]"
    exit 1
fi

# define variables from arguments
base=${1%.*}
twiss=$base.slan
if [ $2 ]; then
    mag=$2
else
    mag=$base.magn
fi
mkdir -p results
mkdir -p results_sigs
mkdir -p OTR

# generate the plot beta vs. norm. centroid emittance ecn
sddsplot \
    -column=s,betac? -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $twiss -graphics=line,vary,scale=2,thickness=2.0 -ylabel='scale=1.5'-xlabel='scale=1.0' -yscale=id=1 \
    -title="betac evolution along the line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results/beta_$base.png \

# generate the plot beta vs. norm. centroid emittance ecn
sddsplot \
    -column=s,charge -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $twiss -graphics=line,vary,scale=2,thickness=2.0 -ylabel='scale=1.5'-xlabel='scale=1.0' -yscale=id=1 \
    -title="charge evolution along the line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results/charge_$base.png \
	

# generate the plot alpha vs s
sddsplot \
    -column=s,alphac? -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $twiss -graphic=line,vary,thickness=2.0 -ylabel='scale=1.5'-xlabel='scale=1.0' -yscale=id=1 \
    -title="alphac evolution along the line: $twiss "\
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results/albpha_$base.png \
		
	
# generate the plot s vs. centroids
sddsplot \
    -column=s,Cx -column=s,Cy -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $twiss -graphic=line,vary,thickness=2.0 -ylabel='scale=1.5'-xlabel='scale=1.0' -yscale=id=1 \
    -title="Centroids evolution along the line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results/Cx_Cy_$base.png \
# generate the plot s vs. emittance	
sddsplot \
    -column=s,en? -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $twiss -graphic=line,vary,thickness=2.0 -ylabel='scale=1.5'-xlabel='scale=1.0' -yscale=id=1 \
    -title="norm. emittance evolution along the line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results/norm_emittance_$base.png \

sddsplot \
    -column=s,en?Ave -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $twiss -graphic=line,vary,thickness=2.0 -ylabel='scale=1.5'-xlabel='scale=1.0' -yscale=id=1 \
    -title="norm. emittance evolution along the line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results/norm_emittance_slice_$base.png \

sddsplot \
    -column=s,ecn?Ave -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $twiss -graphic=line,vary,thickness=2.0 -ylabel='scale=1.5'-xlabel='scale=1.0' -yscale=id=1 \
    -title="norm. emittance evolution along the line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results/centroid_norm_emittance_slice_$base.png \

sddsplot \
    -column=s,SdeltaAve -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $twiss -graphic=line,vary,thickness=2.0 -ylabel='scale=1.5'-xlabel='scale=1.0' -yscale=id=1 \
    -title="norm. emittance evolution along the line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results/energyspread_slice_$base.png \
	
sddsplot \
    -column=s,en? -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $twiss -graphic=line,vary,thickness=2.0 -ylabel='scale=1.5'-xlabel='scale=1.0' -scales=0,13,0,0 -yscale=id=1 \
    -title="norm. emittance evolution along the line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results/norm_emittance2_$base.png \
 		
	
sddsplot \
    -column=s,eta? -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $twiss -graphic=line,vary,thickness=2.0 -ylabel='scale=1.5'-xlabel='scale=1.0' -yscale=id=1 \
	-title="Eta evolution along the E210line: $twiss"\
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results/eta_$base.png \

sddsplot \
    -column=s,duration -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $twiss -graphic=line,vary,thickness=2.0 -ylabel='scale=1.5'-xlabel='scale=1.0' -yscale=id=1 \
	-title="Eta evolution along the E210line: $twiss"\
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results/duration_$base.png \
	
		
sddsplot \
    -column=s,S[xy] -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $base.sig -graphic=line,vary,thickness=2.0 -ylabel='scale=1.0'-xlabel='scale=1.0' -yscale=id=1 \
    -title="BeamSize along the E210line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results/beamSize_$base.png	
	
	sddsplot \
    -column=s,S[xy]p -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $base.sig -graphic=line,vary,thickness=2.0 -ylabel='scale=1.0'-xlabel='scale=1.0' -yscale=id=1 \
    -title="BeamSize along the E210line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results/beamSizediverg_$base.png	

sddsplot \
    -column=s,ecn[xy] -legend=ysymbol -dateStamp -unsup=y -tickSettings=yspacing=30,yfactor='1e9' \
    -zoom=yfac=0.87,qcent=0.53 $base.sig -graphic=line,vary,thickness=2.0 -ylabel='scale=1.0' -yscale=id=1 \
    -title="norm. cen. emittance evolution along the line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results_sigs/normEmitanceC_$base.png	
sddsplot \
    -column=s,en[xy] -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $base.sig -graphic=line,vary,thickness=2.0 -ylabel='scale=1.5'-xlabel='scale=1.0' -tickSettings=size=24 -yscale=id=1 \
    -title="norm. emittance evolution along the line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results_sigs/normEmitance_$base.png

sddsplot \
    -column=s,beta[xy]Beam  -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.87,qcent=0.53 $base.sig -graphic=line,vary,thickness=2.0 -ylabel='scale=1.0'-xlabel='scale=1.0' -yscale=id=1 \
    -title="Beta evolution along the line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results_sigs/BetaSig_$base.png	

sddsplot \
    -column=s,alpha[xy]Beam  -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.70,qcent=0.53 $base.sig -graphic=line,vary,thickness=2.0 -ylabel='scale=1.0'-xlabel='scale=1.0' -yscale=id=1 \
    title="Alpha evolution along the line: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results_sigs/alphaSig_$base.png		

sddsplot \
    -column=s,Sdelta  -legend=ysymbol -dateStamp -unsup=y \
    -zoom=yfac=0.70,qcent=0.53 $base.sig -graphic=line,vary,thickness=2.0 -ylabel='scale=1.0'-xlabel='scale=1.0' -yscale=id=1 \
    -title="energy Spread evolution: $twiss" \
    -column=s,Profile $mag \
    -overlay=xmode=normal,yfactor=0.05,qoffset=-0.46,ycenter,ymode=unit -device=lpng,onwhite -output=results_sigs/energyspread_$base.png		
		
	
#sddsanalyzebeam $base.in start_$base.tmp	
#sddsprintout start_$base.tmp results/twiss_parameter_start_$base.txt -column=betax -column=betay -column=alphax -column=alphay -column=etax -column=etay  -column=enx -column=eny	-column=Sy  -column=Sx -column=Sdelta 
	

#sddsanalyzebeam $base.out end_$base.tmp
#sddsprintout end_$base.tmp results/twiss_parameter_end_$base.txt -column=betax -column=betay -column=alphax -column=alphay -column=etax -column=etay  -column=enx -column=eny -column=Sy  -column=Sx -column=Sdelta


mkdir -p twiss_ascii


sddsprintout $base.slan twiss_ascii/twiss_parameter_slan_$base.txt  -column=s -column=betacx -column=betacy -column=alphacx -column=alphacy -column=etax -column=etay  -column=enx -column=eny -column=duration -noTitle -spreadsheet
sddsprintout $base.sig twiss_ascii/twiss_parameter_sig_$base.txt  -column=betaxBeam -column=betayBeam -column=alphaxBeam -column=alphayBeam -column=enx -column=eny  -column=ecnx -column=ecny -noTitle -spreadsheet
sddsprintout $base.sig twiss_ascii/diverg_sig_$base.txt  -column=s -column=Sxp -column=Syp -noTitle -spreadsheet
sddsprintout $base.twi twiss_ascii/twiss_MaxMinAve_$base.txt -par=betaxMax -par=betaxMin -par=betaxAve -par=betayMax -par=betayMin -par=betayAve -noTitle -spreadsheet

sddsplot $base.out -col=t,p -graph=dots	-device=lpng,onwhite -output=twiss_ascii/LongPhaseSpace_$base.png
sddsplot $base.out -col=x,y -graph=dots	-device=lpng,onwhite -output=twiss_ascii/TrasvXvsY_$base.png	
sddsplot $base.out -col=x,xp -graph=dots	-device=lpng,onwhite -output=twiss_ascii/TrasvXvsXP_$base.png	
sddsplot $base.out -col=y,yp -graph=dots	-device=lpng,onwhite -output=twiss_ascii/TrasvXvsYP_$base.png

sddsplot $base.OTR1 -col=t,p -graph=dots	-device=lpng,onwhite -output=OTR/OTR1LongPhaseSpace_$base.png
sddsplot $base.OTR1 -col=x,y -graph=dots	-device=lpng,onwhite -output=OTR/OTR1TrasvXvsY_$base.png	
#sddsplot $base.OTR1 -col=x,xp -graph=dots	-device=lpng,onwhite -output=OTR/OTR1TrasvXvsXP_$base.png	
#sddsplot $base.OTR1 -col=y,yp -graph=dots	-device=lpng,onwhite -output=OTR/OTR1TrasvXvsYP_$base.png

sddsplot $base.OTR3 -col=t,p -graph=dots	-device=lpng,onwhite -output=OTR/OTR3LongPhaseSpace_$base.png
sddsplot $base.OTR3 -col=x,y -graph=dots	-device=lpng,onwhite -output=OTR/OTR3TrasvXvsY_$base.png
sddsplot $base.OTR3 -col=x,p -graph=dots	-device=lpng,onwhite -output=OTR/OTR3TrasvXvsP_$base.png	
#sddsplot $base.w2 -col=x,xp -graph=dots	-device=lpng,onwhite -output=OTR/w2TrasvXvsXP_$base.png	
#sddsplot $base.w2 -col=y,yp -graph=dots	-device=lpng,onwhite -output=OTR/w2TrasvXvsYP_$base.png

sddsplot $base.OTR5 -col=t,p -graph=dots	-device=lpng,onwhite -output=OTR/OTR5LongPhaseSpace_$base.png
sddsplot $base.OTR5 -col=x,y -graph=dots	-device=lpng,onwhite -output=OTR/OTR5TrasvXvsY_$base.png	
#sddsplot $base.w2 -col=x,xp -graph=dots	-device=lpng,onwhite -output=OTR/w2TrasvXvsXP_$base.png	
#sddsplot $base.w2 -col=y,yp -graph=dots	-device=lpng,onwhite -output=OTR/w2TrasvXvsYP_$base.png

sddsprintout $base.out twiss_ascii/X_$base.txt -column=x -noTitle -spreadsheet
sddsprintout $base.out twiss_ascii/XP_$base.txt -column=xp -noTitle -spreadsheet
sddsprintout $base.out twiss_ascii/Y_$base.txt -column=y -noTitle -spreadsheet
sddsprintout $base.out twiss_ascii/YP_$base.txt -column=yp -noTitle -spreadsheet

sddsprintout transport_line_ideal.in twiss_ascii/BX_$base.txt -column=x
sddsprintout transport_line_ideal.in twiss_ascii/BXP_$base.txt -column=xp
sddsprintout transport_line_ideal.in twiss_ascii/BY_$base.txt -column=y
sddsprintout transport_line_ideal.in twiss_ascii/BYP_$base.txt -column=yp
rm start_$base.tmp
rm end_$base.tmp



	
