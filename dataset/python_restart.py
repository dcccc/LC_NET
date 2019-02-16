import os,string,time


txt_error=open("txt_error.txt","a")

line_num1=os.popen("wc -l Egap_file | awk '{print $1}'").readlines()
line_num2=0


while(line_num1!=line_num2):
	txt=os.popen("ps -axjf  |grep  aflow").readlines()	
	if len(txt)<=2:

		line=os.popen("tail  -n 3 Egap_file").readlines()
		if len(line[0])>10:
			num=string.replace(line[-1][:5],"_","")
			
		elif len(line[1])>10:
			num=string.replace(line[1][:5],"_","")			
		else :
			num=string.replace(line[0][:5],"_","")
		txt_error.write("%s+1  is error\n"  %(num))

		num=int(num)

		line_sed ="for i in range(%d,len(all_list)):"  %(num+2)
		os.popen("sed -i '9c %s' aflow_Egap_download.py"   %(line_sed))

		line_num1=os.popen("wc -l Egap_filefile | awk '{print $1}'").readlines()

		try:
			execfile("aflow_Egap_download.py")
		except Exception,e:
			pass

		line_num2=os.popen("wc -l Egap_filefile | awk '{print $1}'").readlines()


		time.sleep(30)
	
	else:
		time.sleep(30)