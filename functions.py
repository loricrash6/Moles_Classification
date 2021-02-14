import matplotlib.pyplot as plt
import numpy as np
from math import pi

def plot_data(object_to_plot, title):
	plt.figure()
	plt.imshow(object_to_plot)
	plt.title(title)
	plt.show()
	#plt.pause(0.1)

#build a matrix to represent the mole with the color of its centroids
def new_2d_matrix(labels,centroids,im):
	[N1,N2,N3]=im.shape
	new_colors=np.zeros((N1*N2,N3))
	for i in range(len(labels)):
		if(labels[i]==0):
			new_colors[i]=centroids[0]
		elif(labels[i]==1):
			new_colors[i]=centroids[1]
		else:
			new_colors[i]=centroids[2]
	return new_colors

def new_2d_matrix_5centroids(labels,centroids,im):
	[N1,N2,N3]=im.shape
	new_colors=np.zeros((N1*N2,N3))
	for i in range(len(labels)):
		if(labels[i]==0):
			new_colors[i]=centroids[0]
		elif(labels[i]==1):
			new_colors[i]=centroids[1]
		elif(labels[i]==2):
			new_colors[i]=centroids[2]
		elif(labels[i]==3):
			new_colors[i]=centroids[3]
		else:
			new_colors[i]=centroids[4]
	return new_colors

def new_2d_matrix_7centroids(labels,centroids,im):
	[N1,N2,N3]=im.shape
	new_colors=np.zeros((N1*N2,N3))
	for i in range(len(labels)):
		if(labels[i]==0):
			new_colors[i]=centroids[0]
		elif(labels[i]==1):
			new_colors[i]=centroids[1]
		elif(labels[i]==2):
			new_colors[i]=centroids[2]
		elif(labels[i]==3):
			new_colors[i]=centroids[3]
		elif(labels[i]==4):
			new_colors[i]=centroids[4]
		elif(labels[i]==5):
			new_colors[i]=centroids[5]
		else:
			new_colors[i]=centroids[6]
	return new_colors

def find_min_centroid(minim, index_min, centroids):
	for i in range(centroids.shape[0]):
		if(int(centroids[i,0])+int(centroids[i,1])+int(centroids[i,2])<minim):
			minim=int(centroids[i,0])+int(centroids[i,1])+int(centroids[i,2])
			index_min=i
	min_centroid=centroids[index_min,:]
	return min_centroid

#two min centroids when clusters are five
def find_min_centroids(minim, index_min, centroids):
	for i in range(centroids.shape[0]):
		if(int(centroids[i,0])+int(centroids[i,1])+int(centroids[i,2])<minim):
			minim=int(centroids[i,0])+int(centroids[i,1])+int(centroids[i,2])
			index_min=i
	min_centroid=centroids[index_min,:]
	
	second_min=255+255+255
	for j in range(centroids.shape[0]):
		if(int(centroids[j,0])+int(centroids[j,1])+int(centroids[j,2])<second_min and (int(centroids[j,0])+int(centroids[j,1])+int(centroids[j,2])!=minim)):
			second_min=int(centroids[j,0])+int(centroids[j,1])+int(centroids[j,2])
			index_sec_min=j
	second_min_centroid=centroids[index_sec_min,:]

	min_centroids=[min_centroid,second_min_centroid]
	
	return min_centroids
#create a mole where its points are of the color of the darkest centroid, and the others are white
def create_bicolor_image(im_3col, min_centroid, mole):
	for i in range(im_3col.shape[1]-1):
		for j in range(im_3col.shape[0]):
			if(im_3col[i,j,0]==min_centroid[0] and im_3col[i,j,1]==min_centroid[1] and im_3col[i,j,2]==min_centroid[2]):
				mole[i,j,:]=min_centroid[:]
			else:
				mole[i,j,:]=[255,255,255]
#same, but two colors beacause two clusters have been taken
def create_tricolor_image(im_3col, min_centroids, mole):
	min_centroid=min_centroids[0]
	sec_min_centroid=min_centroids[1]

	for i in range(im_3col.shape[1]-1):
		for j in range(im_3col.shape[0]):
			if(im_3col[i,j,0]==min_centroid[0] and im_3col[i,j,1]==min_centroid[1] and im_3col[i,j,2]==min_centroid[2]):
				#mole[i,j,:]=min_centroid[:]
				mole[i,j,:]=[0,0,0] #nero
			elif(im_3col[i,j,0]==sec_min_centroid[0] and im_3col[i,j,1]==sec_min_centroid[1] and im_3col[i,j,2]==sec_min_centroid[2]):
				#mole[i,j,:]=sec_min_centroid[:]
				mole[i,j,:]=[128,128,128]
			else:
				mole[i,j,:]=[255,255,255]
#the two taken clusters are merged, taking the darkest color
def create_bicolor_image_2centroids(im_3col, min_centroids, mole):
	for i in range(im_3col.shape[1]-1):
		for j in range(im_3col.shape[0]):
			if((im_3col[i,j,0]==min_centroids[0][0] and im_3col[i,j,1]==min_centroids[0][1] and im_3col[i,j,2]==min_centroids[0][2]) or (im_3col[i,j,0]==min_centroids[1][0] and im_3col[i,j,1]==min_centroids[1][1] and im_3col[i,j,2]==min_centroids[1][2])):
				mole[i,j,:]=min_centroids[0][:]
			else:
				mole[i,j,:]=[255,255,255]

def remove_image_frames(im):
	ylim_1=int(im.shape[0]*0.1)
	xlim_1=int(im.shape[1]*0.1)
	ylim_2=int(im.shape[0]-ylim_1)
	xlim_2=int(im.shape[1]-xlim_1)
	res_im=np.zeros((im.shape[0],im.shape[1],3),dtype=int)

	for x in range(im.shape[1]-1):
		for y in range(im.shape[0]):
			if x<xlim_1 or y<ylim_1 or x>xlim_2 or y>ylim_2:
				res_im[x,y,:]=[255,255,255]
			else:
				res_im[x,y,:]=im[x,y,:]

	return res_im

def remove_big_image_frames(im):
	ylim_1=int(im.shape[0]*0.2)
	xlim_1=int(im.shape[1]*0.2)
	ylim_2=int(im.shape[0]-ylim_1)
	xlim_2=int(im.shape[1]-xlim_1)
	res_im=np.zeros((im.shape[0],im.shape[1],3),dtype=int)

	for x in range(im.shape[1]-1):
		for y in range(im.shape[0]):
			if x<xlim_1 or y<ylim_1 or x>xlim_2 or y>ylim_2:
				res_im[x,y,:]=[255,255,255]
			else:
				res_im[x,y,:]=im[x,y,:]

	return res_im

def count_mole_points(mole,min_centroid):
	"""evaluate the total number of points in the mole (=darkest cluster)"""
	count=0 
	for i in range(mole.shape[0]):
		for j in range(mole.shape[1]):
			if(mole[i,j,0]==min_centroid[0] and mole[i,j,1]==min_centroid[1] and mole[i,j,2]==min_centroid[2]):
				count+=1
	return count

def obtain_mole_points_coordinates(mole, min_centroid, count_mole):
	"""save the coordinates of all the points of the mole (=darkest cluster)"""
	pos=0
	x_coord=np.zeros((count_mole),dtype=int)
	y_coord=np.zeros((count_mole),dtype=int)

	for i in range(mole.shape[0]):
		for j in range(mole.shape[1]-1):
			if(mole[i,j,0]==min_centroid[0] and mole[i,j,1]==min_centroid[1] and mole[i,j,2]==min_centroid[2]):
				
				x_coord[pos]=j
				y_coord[pos]=i
				pos=pos+1

	#x_coord.sort()
	#y_coord.sort()

	coords=[x_coord,y_coord]
	return coords

def whiten(mole_to_whiten,median_x,median_y,stdev_x,stdev_y,coordinates):

	x_coords=coordinates[0]
	y_coords=coordinates[1]
	whitened=mole_to_whiten.copy()

	for it in range(len(x_coords)):
		x_norm=(x_coords[it]-median_x)/stdev_x
		y_norm=(y_coords[it]-median_y)/stdev_y

		#if this point is in the tail of the normalized gaussian distribution, we make it become white	
		
		if x_norm<=(-1.96) or x_norm>=1.96 or y_norm<=(-1.96) or y_norm>=1.96:
			whitened[int(y_coords[it]),int(x_coords[it]),:]=[255,255,255]
	return whitened

#procedure to remove dark points not part of the mole
def whitening_procedure(mole_to_whiten, min_centroid):
	stdev_x_list=[]
	stdev_y_list=[]
	flag=True
	it=0
	stop=False


	while flag==True:

		count_mole=count_mole_points(mole_to_whiten, min_centroid)
		coords_list=obtain_mole_points_coordinates(mole_to_whiten, min_centroid, count_mole)
		#print(count_mole)

		#calculate median and standard deviation of the darkest cluster (mole) along x and y 
		x_coords=list(coords_list[0])
		x_ord=x_coords.copy()
		x_ord.sort()
		y_coords=list(coords_list[1])

		median_x=np.median(x_ord)
		median_y=np.median(y_coords)

		stdev_x=np.std(x_coords)
		stdev_y=np.std(y_coords)

		stdev_x_list.append(stdev_x)
		stdev_y_list.append(stdev_y)

		#STOPPING CONDITION: we have reduced the stdev by 33% or we are not reducing it (almost) at all 'cause the situation is already good
		if(it!=0):
			#CAPIRE PERCHE' QUESTO +2
			if stdev_y_list[it]+1>stdev_y_list[it-1] or stdev_y_list[0]/stdev_y_list[it]>=1.5:
				stop=True

			elif stdev_x_list[it]+1>stdev_x_list[it-1] or stdev_x_list[0]/stdev_x_list[it]>=1.5:
				stop=True

		if stop==False:
			#print("Median: ("+str(median_x)+","+str(median_y)+")\n")
			#whiten pixels which are on the tails of the distruìibution (i.e. not on the 95% of the points around the median)
			mole_to_whiten=whiten(mole_to_whiten,median_x,median_y,stdev_x,stdev_y,coords_list)
			
		else:
			flag=False
		it+=1

	#stdev_list=[]
	#stdev_list.append(stdev_x_list)
	#stdev_list.append(stdev_y_list)

	#if the first iteration was useless, we remove such values from the list to identify that the mole was already good
	#of course, wrt before it's not taken into account the case in which in one iteration all the desired variance is removed,
	#which would indeed be a fast and effective application of the whitening
	if stdev_x_list[1]+1>stdev_x_list[0] or stdev_y_list[1]+1>stdev_y_list[0]:
		stdev_x_list.pop(1)
		stdev_y_list.pop(1)

	if len(stdev_x_list)!=1:
		#plot_stdev(stdev_x_list, stdev_y_list)
		return mole_to_whiten
	else:
		return False

#whitening for 5 clusters procedure
def re_whiten(mole_to_whiten,min_centroid):

	whitened=mole_to_whiten.copy()
	count_mole=count_mole_points(mole_to_whiten, min_centroid)
	coords_list=obtain_mole_points_coordinates(mole_to_whiten, min_centroid, count_mole)

	#calculate median and standard deviation of the darkest cluster (mole) along x and y 
	x_coords=list(coords_list[0])
	x_ord=x_coords.copy()
	x_ord.sort()
	y_coords=list(coords_list[1])

	median_x=np.median(x_ord)
	median_y=np.median(y_coords)

	stdev_x=np.std(x_coords)
	stdev_y=np.std(y_coords)

	for it in range(len(x_coords)):
		x_norm=(x_coords[it]-median_x)/stdev_x
		y_norm=(y_coords[it]-median_y)/stdev_y

		if x_norm<=(-2.25) or x_norm>=2.25 or y_norm<=(-2.25) or y_norm>=2.25: 
				whitened[int(y_coords[it]),int(x_coords[it]),:]=[255,255,255]

	return whitened

#whitening for 7 clusters procedure
def re_whiten_sev(mole_to_whiten,min_centroid):

	whitened=mole_to_whiten.copy()
	count_mole=count_mole_points(mole_to_whiten, min_centroid)
	coords_list=obtain_mole_points_coordinates(mole_to_whiten, min_centroid, count_mole)

	#calculate median and standard deviation of the darkest cluster (mole) along x and y 
	x_coords=list(coords_list[0])
	x_ord=x_coords.copy()
	x_ord.sort()
	y_coords=list(coords_list[1])

	median_x=np.median(x_ord)
	median_y=np.median(y_coords)

	stdev_x=np.std(x_coords)
	stdev_y=np.std(y_coords)

	for it in range(len(x_coords)):
		x_norm=(x_coords[it]-median_x)/stdev_x
		y_norm=(y_coords[it]-median_y)/stdev_y

		if x_norm<=(-2.75) or x_norm>=2.75 or y_norm<=(-2.75) or y_norm>=2.75:
				whitened[int(y_coords[it]),int(x_coords[it]),:]=[255,255,255]

	return whitened


def blurring_procedure(grouping_dim,threshold,mole_to_blur, min_centroid):
	"""to remove single sparse outliers"""
	for i in range(0,mole_to_blur.shape[0]-grouping_dim,3):
		for j in range(0, mole_to_blur.shape[1]-grouping_dim, grouping_dim):
			group=np.zeros((grouping_dim,grouping_dim,3),dtype=np.uint8)
			for it1 in range(0,grouping_dim):
				for it2 in range(0,grouping_dim):
 					group[it1,it2,:]=mole_to_blur[i+it1,j+it2,:]
			counter=count_mole_points(group, min_centroid)
			if counter<=threshold:
				for it1 in range(0,grouping_dim):
					for it2 in range(0,grouping_dim):
						mole_to_blur[i+it1,j+it2,:]=[255,255,255]
	return mole_to_blur

def plot_stdev(stdev_x_list, stdev_y_list):
	stdev_x=np.zeros(len(stdev_x_list),dtype=float)
	stdev_y=np.zeros(len(stdev_y_list),dtype=float)

	for it in range(len(stdev_x_list)):
		stdev_x[it]=stdev_x_list[it]
		stdev_y[it]=stdev_y_list[it]

	dim=len(stdev_x_list)
	fig,axarr=plt.subplots(1,2,squeeze=False)
	ax=np.arange(dim)
	fig.suptitle('Standard deviations of mole\'s points', fontsize=20)
	#l, h = fig.get_size_inches()
	#zoom = 2
	#fig.set_size_inches(l * zoom, h * zoom)
	axarr[0][0].plot(ax,stdev_x, 'tab:red')
	axarr[0, 0].set_title('Stdev_x', fontsize=10)
	axarr[0][1].plot(ax,stdev_y, 'tab:red')
	axarr[0, 1].set_title('Stdev_y', fontsize=10)
	plt.show()

def get_rect(mole, min_centroid):
	count_mole=count_mole_points(mole,min_centroid)
	#print(count_mole)
	list_coords=obtain_mole_points_coordinates(mole, min_centroid, count_mole)

	list_x=list(list_coords[0])
	list_y=list(list_coords[1])

	list_x.sort()

	median_x=np.median(list_x)
	median_y=np.median(list_y)
	
	x_left=int(np.amin(list_coords[0]))
	x_right=int(np.amax(list_coords[0]))
	y_top=int(np.amin(list_coords[1]))
	y_bottom=int(np.amax(list_coords[1]))

	rect=[]
	rect.append(x_left)
	rect.append(x_right)
	rect.append(y_bottom)
	rect.append(y_top)
	rect.append(median_x)
	rect.append(median_y)
	return rect

def draw_rect(rect,mole):

	x_left=rect[0]
	x_right=rect[1]
	y_bottom=rect[2]
	y_top=rect[3]
	median_x=rect[4]
	median_y=rect[5]

	vector_len=np.arange(x_left,x_right)
	heigh=np.ones(len(vector_len))*y_top
	low=np.ones(len(vector_len))*y_bottom

	vector_height=np.arange(y_top,y_bottom)
	right=np.ones(len(vector_height))*x_right
	left=np.ones(len(vector_height))*x_left

	plt.figure()
	plt.title("Identified mole should be in the rectangular section")
	plt.plot(vector_len,low,'r-',vector_len,heigh,'r-')
	plt.plot(left,vector_height,'r-',right,vector_height,'r-')
	plt.plot(median_x,median_y,'ro')
	plt.imshow(mole)
	plt.show()

def save_rect(rect,mole,title):
	x_left=rect[0]
	x_right=rect[1]
	y_bottom=rect[2]
	y_top=rect[3]
	median_x=rect[4]
	median_y=rect[5]

	vector_len=np.arange(x_left,x_right)
	heigh=np.ones(len(vector_len))*y_top
	low=np.ones(len(vector_len))*y_bottom

	vector_height=np.arange(y_top,y_bottom)
	right=np.ones(len(vector_height))*x_right
	left=np.ones(len(vector_height))*x_left

	fig=plt.figure()
	plt.title("Identified mole should be in the rectangular section")
	plt.plot(vector_len,low,'r-',vector_len,heigh,'r-')
	plt.plot(left,vector_height,'r-',right,vector_height,'r-')
	plt.plot(median_x,median_y,'ro')
	plt.imshow(mole)
	title="plots/"+title[11:]
	fig.savefig(title)
	plt.show()

#image to show to the user in order to choose if one cluster is enough to describe the mole or the two darkest clusters are needed
def plot_3col_data(mole,title,im):
	fig,axarr=plt.subplots(1,2,squeeze=False)
	fig.suptitle('Two clusters vs true image', fontsize=30)
	axarr[0][0].imshow(mole)
	axarr[0][1].imshow(im)
	plt.show()

def find_perimeter(mole):
	mole_per=mole.copy()
	perim_pix=[]
	for i in range(mole.shape[1]-1):
		for j in range(mole.shape[0]):
			if (mole[i,j,0]!=255 and mole[i-1,j,0]==255) or (mole[i,j,0]!=255 and mole[i+1,j,0]==255) or (mole[i,j,0]!=255 and mole[i,j-1,0]==255) or (mole[i,j,0]!=255 and mole[i,j+1,0]==255):
				mole_per[i,j,:]=[255,0,0]
				ob={"xcoord":i,"ycoord":j}
				perim_pix.append(ob)
	to_return=[]
	to_return.append(mole_per)
	to_return.append(perim_pix)

	return to_return

def find_area(mole):
	a=0
	for i in range(mole.shape[1]-1):
		for j in range(mole.shape[0]):
			if mole[i,j,0]!=255:
				a=a+1
	return a

#functions to analyze the perimeter (see the main.py for description)
def greening(mole):
	green_mole=mole.copy()
	for it in range(mole.shape[0]):
		black_list=[]
		for it2 in range(mole.shape[1]-1):
			if mole[it2,it,0]!=255:
				black_list.append(it2)
		if len(black_list)>0:
		#extract limits for black points on the considered row
			left=int(np.amin(black_list))
			right=int(np.amax(black_list))

			for it3 in range(mole.shape[1]-1):
				if it3>left and it3<right and mole[it3,it,0]==255:
					green_mole[it3,it,:]=[0,255,0]

	count_green=0
	for it in range(mole.shape[0]):
		for it2 in range(mole.shape[1]-1):
			if green_mole[it2,it,0]==0 and green_mole[it2,it,1]==255 and green_mole[it2,it,2]==0:
				count_green=count_green+1
	#print("Green pixels: "+str(count_green))
	return green_mole

def degulf(mole):
	
	count=1 #fake initialization

	#go on until no more green pixels are removable
	while count>0:
		count=degulf_procedure(mole)

	return mole


def degulf_procedure(mole):

	count=0 #count of removed green pixels

	for it in range(mole.shape[0]):
		#check 4 neighbors of each green pixel: if at least one is white, we are not inside the mole, so the green point must be whitened
		#from left to right and viceversa!
		
		for it2 in range(mole.shape[1]-1):
			if mole[it2,it,0]==0 and mole[it2,it,1]==255 and mole[it2,it,2]==0:
				if (mole[it2+1,it,0]==255 and mole[it2+1,it,1]==255 and mole[it2+1,it,2]==255) or (mole[it2,it+1,0]==255 and mole[it2,it+1,1]==255 and mole[it2,it+1,2]==255) or (mole[it2-1,it,0]==255 and mole[it2-1,it,1]==255 and mole[it2-1,it,2]==255) or (mole[it2,it-1,0]==255 and mole[it2,it-1,1]==255 and mole[it2,it-1,2]==255):
					count=count+1
					mole[it2,it,:]=[255,255,255]

		for it3 in range(mole.shape[1]-2,0,-1):
			if mole[it3,it,0]==0 and mole[it3,it,1]==255 and mole[it3,it,2]==0:
				if (mole[it3+1,it,0]==255 and mole[it3+1,it,1]==255 and mole[it3+1,it,2]==255) or (mole[it3,it+1,0]==255 and mole[it3,it+1,1]==255 and mole[it3,it+1,2]==255) or (mole[it3-1,it,0]==255 and mole[it3-1,it,1]==255 and mole[it3-1,it,2]==255) or (mole[it3,it-1,0]==255 and mole[it3,it-1,1]==255 and mole[it3,it-1,2]==255):
					count=count+1
					mole[it3,it,:]=[255,255,255]

	#print("Count: "+str(count))

	return count

def color_lakes(mole_init,min_centroid):
	mole=mole_init.copy()
	for it in range(mole.shape[0]):
		for it2 in range(mole.shape[1]-1):
			if mole[it2,it,0]==0 and mole[it2,it,1]==255 and mole[it2,it,2]==0:
				mole[it2,it,:]=min_centroid[:]

	return mole

#functionn for symmetry analysis
def rotate(per_coords,mole,min_centroid):
	
	mole_count=count_mole_points(mole,min_centroid)
	coords_list=obtain_mole_points_coordinates(mole,min_centroid,mole_count)
	x_coords=list(coords_list[0])
	#x_ord=x_coords.copy()
	#x_ord.sort()
	y_coords=list(coords_list[1])
	x_coords=x_coords-np.mean(x_coords)
	y_coords=y_coords-np.mean(y_coords)
	mole_coords=np.vstack([x_coords,y_coords])

	cov=np.cov(mole_coords)
	evals,evecs=np.linalg.eig(cov)
	sort_indices=np.argsort(evals)[::-1]
	x_v1, y_v1=evecs[:,sort_indices[0]]  # Eigenvector with largest eigenvalue
	x_v2, y_v2=evecs[:,sort_indices[1]]

	scale = 20
	plt.plot(x_coords, y_coords, 'b.')
	plt.plot([x_v1*-scale*2, x_v1*scale*2],[y_v1*-scale*2, y_v1*scale*2], color='red')
	plt.plot([x_v2*-scale, x_v2*scale],[y_v2*-scale, y_v2*scale], color='green')
	plt.axis('equal')
	#plt.gca().invert_yaxis()  # Match the image system with origin at top left
	plt.show()

	theta=np.arctan((x_v1)/(y_v1))
	rotation_mat=np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	transformed_mat=rotation_mat*mole_coords
	# plot the transformed blob
	x_transformed,y_transformed=transformed_mat.A
	for it in range(len(x_transformed)):
		x_transformed[it]=int(x_transformed[it])
	for it in range(len(y_transformed)):
		y_transformed[it]=int(y_transformed[it])
	plt.plot(x_transformed,y_transformed,'g.')
	#plt.show()
	
	#simmetry towards vertical axis
	y_min=int(np.amin(y_transformed))
	y_max=int(np.amax(y_transformed))

	ics=[]
	ips=[]
	for it in range(y_min,y_max):
		#pick all the points of a row
		row=[]
		for it2 in range(len(y_transformed)):
			if y_transformed[it2]==it:
				row.append(x_transformed[it2])
		#we have all the x-coords of the points of row num. it
		#evaluate median and plot it
		if len(row)>0:
			m=np.median(row)
			ics.append(m)
			ips.append(it)
	plt.plot(ics,ips,'r.')
	#plt.show()

	d=0
	for it in range(len(ics)):
		d=d+abs(ics[it])
	#print("d: "+str(d))
	#print("num. rows: "+str(y_max+abs(y_min)))
	simx=round(d/((y_max+abs(y_min))),2)  

	#simmetry towards horizontal axis
	x_min=int(np.amin(x_transformed))
	x_max=int(np.amax(x_transformed))

	ics=[]
	ips=[]
	for it in range(x_min,x_max):
		#pick all the points of a row
		col=[]
		for it2 in range(len(x_transformed)):
			if x_transformed[it2]==it:
				col.append(y_transformed[it2])
		#we have all the x-coords of the points of col num. it
		#evaluate median and plot it
		if len(col)>0:
			m=np.median(col)
			ips.append(m)
			ics.append(it)
	plt.plot(ics,ips,'b.')
	plt.show()

	d=0
	for it in range(len(ips)):
		d=d+abs(ips[it])
	#print("d: "+str(d))
	#print("num. cols: "+str(x_max+abs(x_min)))
	simy=round(d/(x_max+abs(x_min)),2)

	#print("Simmetry wrt principal axis and its orthogonal one: "+str(round((simx+simy)/2,2)))
	#or we could take the max btw simx and simy..? 

	return round((simx+simy)/2,2)

def write_results(r,sim,per_file,sym_file):
	#per=open(per_file,'a')
	#per.write(r)
	#per.write("\n")

	sym=open(sym_file,'a')
	sym.write(sim)
	sym.write("\n")

	#per.close()
	sym.close()

		
