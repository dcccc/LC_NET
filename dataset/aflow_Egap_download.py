from aflow import *
import string,time,os

all_list=search(catalog="ICSD").filter(K.Egap>-1.0).filter(K.density>0.5).filter(K.delta_electronic_energy_convergence<0.0001).filter(K.dft_type=="PAW_PBE")


Egap_file=open("Egap_file","a")

for i in range(len(all_list)):
	entry=all_list[i]

	file=entry.files
	aa=filter(lambda x : "cif" in x ,list(file))
	# print(aa)
	aa=aa[0]


	if "CONTCAR.relax.qe" in list(file):
		text=string.replace(file["CONTCAR.relax.qe"](),"!","\n!")
		file_qe=aa[:-4]+".qe"
	elif "CONTCAR.relax.vasp" in list(file):
		text=file["CONTCAR.relax.qe"]()
		file_qe=aa[:-4]+".vasp"
	else:
		text=file["CONTCAR.relax"]()
		file_qe=aa[:-4]+".vasp"
	
	n=i//10000
	
	if not os.path.isdir():
		os.mkdir(str(n))	
	
	qe_file=open(str(i)+"__"+file_qe,"a")
	qe_file.write(text)
	qe_file.close()

	Egap=entry.Egap
	Egap_type=entry.Egap_type

	text=str(i)+"__"+"%-40s    %-10s    %-f \n"  %(aa[:-4]  ,Egap_type[:-2],  Egap )
	Egap_file.write(text)

	if i%50==0:
		time.sleep(1)
		Egap_file.flush()
