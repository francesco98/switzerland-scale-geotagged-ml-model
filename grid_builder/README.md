# Description
Run the scripts here in the following order
1) flickr_search_images.py: searches for new image id's om flicker, updates the file output/flickr_images.csv
2) download_images.py: download the images into the data_dir folder
3) validate_images.py: checks the image files, runs all transformations, updates the files output/flickr_images_excluded.csv and output/flickr_images_validated.csv
4) GridBuilder.py: creates the adaptive grid, output/grid_cellIds.csv and output/grid_polygons.csv
5) lable_imgages.py: creates the adaptive labels, creates the files output/flickr_images_label.csv and output/flickr_images_grid.csv