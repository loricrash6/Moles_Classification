import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import functions as f

#PRELIMINARY OPERATIONS

#creating a list with all the images names
files_in=[]
base_1="moles/low_risk_"
base_2="moles/medium_risk_"
base_3="moles/melanoma_"
for i in range(1,12):
	it=base_1+str(i)+".jpg"
	files_in.append(it)
for i in range(1,17):
	it=base_2+str(i)+".jpg"
	files_in.append(it)
for i in range(1,28):
	it=base_3+str(i)+".jpg"
	files_in.append(it)

#code to un-comment to perform the procedure on one isolated mole
file_in=files_in[32]

files_in=[]
files_in.append(file_in)

#procedure for one mole at a time
for item in range(len(files_in)):
	filein=files_in[item]

	im=mpimg.imread(filein)
	#f.plot_data(im, 'Original image')


	#applying kmeans algortihm
	kmeans=KMeans(n_clusters=3, random_state=0)
	[N1,N2,N3]=im.shape
	im_2D=im.reshape((N1*N2,N3))
	kmeans.fit(im_2D)

	#calculating the centroids of the clusters
	centroids=kmeans.cluster_centers_.astype('uint8')
	#calculating the labels of each pixel
	labels=kmeans.labels_

	#2D matrix with new colors based on lables and centroids
	#we substitute the colors with the centroid color of the pixel label
	new_colors=f.new_2d_matrix(labels,centroids,im)

	#making the new matrix 3D (simple reshaping of the 2D one)
	im_3col=new_colors.reshape((N1,N2,N3))
	im_3col=im_3col.astype('uint8')
	#f.plot_data(im_3col, "Image with centroids colors")


	#find the minimum centroid corresponding to the darkest color (=the mole)
	min_col=255+255+255 #Fake initialization (this is actually the more lightful color --> white)
	index_min=3 #Impossible to have such an index (it goes from 0 to 2)
	min_centroid=f.find_min_centroid(min_col, index_min, centroids)

	#creating an image where the mole has the darkest color of the minimum centroid and white color around
	mole=np.zeros((N1,N2,N3),dtype=float)
	f.create_bicolor_image(im_3col, min_centroid, mole)
	mole=mole.astype('uint8')
	#f.plot_data(mole, "Image with 2 colors (white around the the darkest centroid)")

	#remove 10% border
	#original_mole_s=f.remove_image_frames(im)
	#f.plot_data(original_mole_s,"Original image with frames removed")
	mole_s=f.remove_image_frames(mole)
	#f.plot_data(mole_s,"2-color image with frames removed")

	#outliers must be taken away from the mole cluster
	#copy for outliers removal
	mole_to_whiten=mole_s.copy()

	#firstly whitening: points far from the median are removed
	whitened_mole=f.whitening_procedure(mole_to_whiten, min_centroid)

	if isinstance(whitened_mole,bool):
		print("The mole was ok, no need of proceedings for massive outliers elimination.")
		whitened_mole=mole_to_whiten
		#blurring to eliminate isolated outliers
		final_mole=f.blurring_procedure(5,3,whitened_mole,min_centroid)
		#f.plot_data(final_mole, "Given mole (blurred for isolated outliers elimination).")
		#f.plot_data(final_mole, "More after denoising")
	else:
		#f.plot_data(whitened_mole, "Mole after outliers removal (1st step)")
		#blurring
		final_mole=f.blurring_procedure(5,3,whitened_mole,min_centroid)
		#f.plot_data(final_mole, "More after blurring (2nd step)")
		#f.plot_data(final_mole, "More after denoising")

	#rectangle to see if the mole is correctly identified
	rect=f.get_rect(final_mole,min_centroid)
	#f.draw_rect(rect,final_mole)
	f.draw_rect(rect,im)

	flag=True
	five_c=False #if true, the procedure must be repeated using 5 clusters instead of 3 (for difficult moles where 3 are not sufficient)

	while flag==True:
		cmd=input("Is the mole well delimited by the red rectangle? If not, a more sophisticated procedure will be adopted. [y/n] \n ")
		if cmd=="y":
			flag=False
			#f.save_rect(rect,im,filein) to save the image (not needed)
			print("Ok, done!")
		elif cmd=="n":
			five_c=True
			flag=False
		else:
			print("Not recognized command!")

	if five_c==True:

		#re-apply k-means with 5 clusters
		kmeans=KMeans(n_clusters=5, random_state=0)
		[N1,N2,N3]=im.shape
		im_2D=im.reshape((N1*N2,N3))
		kmeans.fit(im_2D)

		#calculating the centroids of the clusters
		centroids=kmeans.cluster_centers_.astype('uint8')
		#calculating the labels of each pixel
		labels=kmeans.labels_

		#2D matrix with new colors based on lables and centroids
		#we substitute the colors with the centroid color of the pixel label
		new_colors=f.new_2d_matrix_5centroids(labels,centroids,im)

		#making the new matrix 3D (simple reshaping of the 2D one)
		im_5col=new_colors.reshape((N1,N2,N3))
		im_5col=im_5col.astype('uint8')
		#f.plot_data(im_5col, "Image with centroids colors")

		#find the minimum centroids corresponding to the two darkest colors 
		min_col=255+255+255 #fake initialization 
		index_min=5 #impossible index (it goes from 0 to 4)

		min_centroids=f.find_min_centroids(min_col, index_min, centroids)
		min_centroid=min_centroids[0]
		#creating an image where the mole has the two darkest colors of the two minimum centroids and white color around
		mole=np.zeros((N1,N2,N3),dtype=float)
		f.create_tricolor_image(im_5col, min_centroids, mole)
		mole=mole.astype('uint8')
		#f.plot_3col_data(mole, "Image with 5 colors",im)

		flag=True
		two_c=False
		sev_c=False

		while flag==True:
			#three possibilities: 5 clusters taking only the darkest one, 5 clusters taking the two darkest ones, or repeat with 7 clusters
			print("Type 'b' if the mole is well defined by the black area, 'g' if also the grey area is needed to describe well the mole or 'r' is even the black area is too dispersed. \n")
			f.plot_3col_data(mole, "Image with 3 colors",im)
			cmd=input("Command: ")
			if cmd=="b":
				flag=False
			elif cmd=="g":
				two_c=True
				flag=False
			elif cmd=="r":
				sev_c=True #repeat with 7 cluster (very difficult mole)
				flag=False
			else:
				print("Not recognized command!")

		if sev_c==False:

			mole_new=np.zeros((N1,N2,N3),dtype=float)
			if two_c==False: #only darkest cluster
				f.create_bicolor_image(im_5col, min_centroid, mole_new)
			elif two_c==True: #two clusters
				f.create_bicolor_image_2centroids(im_5col, min_centroids, mole_new)

			mole_nf=f.remove_image_frames(mole_new)

			mole=mole_nf.astype('uint8')
			#f.plot_data(mole, "Image with 2 colors (white around the mole)")

			#light blurring
			blurred_mole=f.blurring_procedure(5,3,mole,min_centroid)
			mole=blurred_mole

			#repeated whitening with user interaction to stop
			Flag=True
			count=1
			while Flag==True:
				if count==1:
					final_mole=mole
					rect=f.get_rect(final_mole,min_centroid)
					#f.draw_rect(rect,final_mole)
					f.draw_rect(rect,im)
				else:
					if count==2:
						#blurring
						blurred_mole=f.blurring_procedure(15,10,mole,min_centroid)
						mole=blurred_mole

					whitened_mole=f.re_whiten(mole,min_centroid)
					final_mole=whitened_mole
					rect=f.get_rect(final_mole,min_centroid)
					#f.draw_rect(rect,final_mole)
					f.draw_rect(rect,im)
					#f.plot_data(whitened_mole, "Mole after outliers removal - step"+str(count))

				c=input("Type 'no' if the rectangle still does not fit well the mole: \n")
				if c=="no":
					mole=final_mole
					count=count+1
				else:
					Flag=False
					#f.save_rect(rect,im,filein)
					print("Ok, done!")
		else:
			
			#re-apply k-means with 7 clusters
			kmeans=KMeans(n_clusters=7, random_state=0)
			[N1,N2,N3]=im.shape
			im_2D=im.reshape((N1*N2,N3))
			kmeans.fit(im_2D)

			#calculating the centroids of the clusters
			centroids=kmeans.cluster_centers_.astype('uint8')
			#calculating the labels of each pixel
			labels=kmeans.labels_

			#2D matrix with new colors based on lables and centroids
			#we substitute the colors with the centroid color of the pixel label
			new_colors=f.new_2d_matrix_7centroids(labels,centroids,im)

			#making the new matrix 3D (simple reshaping of the 2D one)
			im_7col=new_colors.reshape((N1,N2,N3))
			im_7col=im_7col.astype('uint8')
			f.plot_data(im_7col, "Image with centroids colors")

			#find the minimum centroids corresponding to the two darkest colors 
			min_col=255+255+255 #fake initialization 
			index_min=7 #impossible index (it goes from 0 to 6)

			min_centroids=f.find_min_centroids(min_col, index_min, centroids)
			min_centroid=min_centroids[0]
			#creating an image where the mole has the two darkest colors of the two minimum centroids and white color around
			mole=np.zeros((N1,N2,N3),dtype=float)
			f.create_tricolor_image(im_7col, min_centroids, mole)
			mole=mole.astype('uint8')
			#f.plot_3col_data(mole, "Image with 3 colors",im)

			flag=True
			two_c=False

			while flag==True:
				#selection of one or two clusters to describe the mole
				print("Type 'b' if the mole is well defined by the black area, 'g' if also the grey area is needed to describe well the mole. \n")
				f.plot_3col_data(mole, "Image with 3 colors",im)
				cmd=input("Command: ")
				if cmd=="b":
					flag=False
				elif cmd=="g":
					two_c=True
					flag=False
				else:
					print("Not recognized command!")

			

			mole_new=np.zeros((N1,N2,N3),dtype=float)
			if two_c==False:
				f.create_bicolor_image(im_7col, min_centroid, mole_new)
			elif two_c==True:
				f.create_bicolor_image_2centroids(im_7col, min_centroids, mole_new)

			mole_nf=f.remove_big_image_frames(mole_new)
			mole=mole_nf.astype('uint8')
			#f.plot_data(mole, "Image with 2 colors (white around the mole)")
			
			#light blurring
			blurred_mole=f.blurring_procedure(5,3,mole,min_centroid)
			mole=blurred_mole

			#whithening like with 5 clusters
			Flag=True
			count=1
			while Flag==True:
				if count==1:
					final_mole=mole
					rect=f.get_rect(final_mole,min_centroid)
					#f.draw_rect(rect,final_mole)
					f.draw_rect(rect,im)
				else:
					if count==2:
						#blurring
						blurred_mole=f.blurring_procedure(5,3,mole,min_centroid)
						mole=blurred_mole

					whitened_mole=f.re_whiten_sev(mole,min_centroid)
					final_mole=whitened_mole
					rect=f.get_rect(final_mole,min_centroid)
					f.draw_rect(rect,final_mole)
					f.draw_rect(rect,im)

				c=input("Type 'no' if the rectangle still does not fit well the mole: \n")
				if c=="no":
					mole=final_mole
					count=count+1
				else:
					f.save_rect(rect,im,filein)
					print("Ok, done!")
					Flag=False

	#we now have the mole, we want to evaluate its perimeter
	#to do so, white holes inside the black area need to be filled!

	#for each row, make the white points btw the two extreme black points become green
	green_mole=f.greening(final_mole)
	#f.plot_data(green_mole,"Greening")

	#now green points in contact with white points must be whitened, to keep only "lakes" but not "gulfs" coloroured by green
	laked_mole=f.degulf(green_mole)
	#f.plot_data(laked_mole,"Degulfing")

	#green points are now only those to be coloured with the mole's color
	definite_mole=f.color_lakes(laked_mole,min_centroid)
	#f.plot_data(definite_mole,"Color lakes")

	#we have our definite mole
	#perimeter and area calculation of the mole
	res=f.find_perimeter(definite_mole)
	per_im=res[0]
	per_coords=res[1]
	p=len(per_coords)
	#f.plot_data(per_im, "Mole with perimeter highlighted in red")
	print("Mole perimeter: "+str(p))

	a=f.find_area(final_mole)
	print("Mole area: "+str(a))

	#found perimeter must be compared to the one of a circle having the same area of the mole
	pc=int(np.sqrt(a/np.pi)*2*np.pi)
	print("Perimeter of equivalent circle: "+str(pc))

	#ratio
	r=round(p/pc,3)
	print("Perimeters ratio: "+str(r))

	#symmetry evaluation
	sim=f.rotate(per_coords,definite_mole,min_centroid)
	print("Simmetry wrt principal axis and its orthogonal one: "+str(sim))

	per_file="perimeters.txt"
	sym_file="symmetries.txt"
	f.write_results(str(r),str(sim),per_file,sym_file)

