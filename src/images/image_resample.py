# Import libraries

import os
import numpy as np
import argparse

from glob import glob
from tqdm import tqdm
import nibabel
import nibabel.processing


def average_voxel_calculator(nifti_dir):

    voxel_sizes = []

    for i in glob(nifti_dir+"\*\*"):
        img = nibabel.load(i)
        voxel_sizes.append(img.header.get_zooms())

    return list(np.mean(voxel_sizes, axis=0))

def Resample2BC(img_path, output_path, method1='cubic', method2='neighbours', voxel=1, smooth_if_larger = False, smooth_sd = 0.5, dirfilter = None, force = False, copyX = False, skipto = False):

    # img_path is an existing directory with data formatted for TexLAB
    # i.e. one directory per case, each containing case.nii(.gz) and one or more
    # segmentation files: somemask.nii(.gz) someothermask.nii(.gz) with one mask
    # label per segmentation file
    # method1 = interpolation method used for images ('cubic')
    # method2 = interpolation method used for masks ('neighbours')
    #   interpolation methods are 'neighbours', 'trilinear', 'cubic'
    # voxel = resliced voxel dimensions
    #   voxel = [sx, sy, sz]
    #   voxel = 1, 2, 3, 4 or 5 implies voxel = [1, 1, voxel]
    # smooth_if_larger = flag for including smoothing step before interpolation to larger voxels
    # smooth_sd = standard deviation of Gaussian smoothing kernel in units of resliced voxel width
    # dirfilter specifies a suffix for filtering image directories (None = no filtering by default)
    # force = True enables overwrite of existing images
    # copyX = True copies qform and sform from image to all segmentations
    
    # Get slice-thickness for resampling
    # In-plane is 1x1 unless specified explicitly e.g. [0.5, 0.5, 1.25]
    if isinstance(voxel, list):
        voxel_shape=voxel 
    elif voxel>5: 
        print('voxel default ranges 1:5, alternatively provide list of voxel dims i.e voxel=[1,1,6]')
        return
    else:
        voxel_shape=[1, 1,voxel]
    print(voxel_shape)
        
    # Resampling types
    order_dict = {'neighbours' : 0, 'trilinear' : 1, 'cubic' : 3}   
    if method1 in order_dict:
        order1 = order_dict[method1]
    else:
        print('error - method1 must be one of', list(order_dict.keys()))          
        return
    if method2 in order_dict:
        order2 = order_dict[method2]
    else:
        print('error - method2 must be one of', list(order_dict.keys()))          
        return
    
    # Don't allow output over input 
    if img_path == output_path:
        print('error - output_path must be different to img_path', img_path)
        return
       
    # Check main image directory exists
    if not os.path.isdir(img_path):
        print('error -', img_path, 'directory cannot be found')
        return
    
    # Make main output directory if necessary
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # Get (filtered) list of image directories
    input_list=os.listdir(img_path)
    input_list = [f for f in input_list if os.path.isdir(os.path.join(img_path, f))]
    if not (dirfilter is None):
        input_list = [f for f in input_list if str.endswith(f, dirfilter)]
           
    # Main loop over image directories
    for im in input_list:
        
        if skipto != False:
            if skipto == im:
                skipto = False
            else:
                print(' skipping case', im)
                continue
        
        print(im)
        
        # Get input directory and filename
        input_dir = os.path.join(img_path, im)
        this_input_file = im+'.nii'
        input_file = os.path.join(input_dir, this_input_file)
        if not os.path.isfile(input_file):
            this_input_file = im+'.nii.gz'
            input_file = os.path.join(input_dir, this_input_file)
            if not os.path.isfile(input_file):
                print('error - image file', im, 'not found so skipping')
                continue
         
        # Check output exists and forced condition
        output_dir = os.path.join(output_path, im)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        outputfile = os.path.join(output_dir, this_input_file)
        if os.path.isfile(outputfile) and not force:
            print(outputfile, 'exists so skipping')
            continue                

        # Load input image
        try:
            input_img = nibabel.load(input_file)
        except nibabel.filebasedimages.ImageFileError:
            print('error opening', input_file, 'so skipping')
            continue
        
        # Apply smoothing if requested when output voxel dimension > input
        if smooth_if_larger:
            # Get rid of empty image dimensions which crash smooth_image
            input_img = nibabel.squeeze_image(input_img);
            ipaffine = input_img.affine
            pixdim = input_img.header.get_zooms()
            nd = len(voxel_shape);
            pixdim = pixdim[0:nd]
            sds = np.array([0.0 if pixdim[i] > voxel_shape[i] else smooth_sd*voxel_shape[i] for i in range(nd)])
            fwhm = nibabel.processing.sigma2fwhm(sds)
            print(sds, fwhm, pixdim)
            input_img = nibabel.processing.smooth_image(input_img, fwhm)
            
        # Apply resampling
        pixdim = input_img.header.get_zooms()
        nd = len(voxel_shape);
        voxel_shape1 = voxel_shape[0:nd]
        if (nd == 2) and (len(pixdim) > 2):
            voxel_shape1 = voxel_shape + [pixdim[2]]
        resampled = nibabel.processing.resample_to_output(input_img, voxel_shape1, order=order1)

        # Save in new directory tree
        nibabel.save(resampled, outputfile)
        opaffine = resampled.affine
        
        if copyX:
            print('Copying resampled image qform and sform')
            imgqform = resampled.get_qform(coded=True)
            imgsform = resampled.get_sform(coded=True)
            sformaffine = imgsform[0];
            sformcode = int(imgsform[1])
            qformaffine = imgqform[0]
            qformcode = int(imgqform[1])
        
        # Deal with segmentations in same directory
        allfiles = os.listdir(input_dir)
        for thisfile in allfiles:
            # Already processed image
            thisfullfile = os.path.join(input_dir, thisfile)
            if thisfullfile == input_file:
                continue
            if not (str.endswith(thisfile, '.nii') or str.endswith(thisfile, '.nii.gz')):
                continue
            # Check if image exists and overwrite enabled
            outputfile = os.path.join(output_dir, thisfile)
            if os.path.isfile(outputfile) and not force:
                print(outputfile, 'exists so skipping')
                continue                

            # Load input image
            try:
                input_seg = nibabel.load(thisfullfile)
            except nibabel.filebasedimages.ImageFileError:
                print('error opening', thisfile, 'so skipping')
                continue
                        
            # Apply smoothing if requested when output voxel dimension > input
            if smooth_if_larger:
                input_seg = nibabel.squeeze_image(input_seg);
                ipheader = input_seg.header
                ipaffine = input_seg.affine
                sdata = input_seg.get_fdata()
                sdata = 100.0*np.asarray(sdata)
                sdata = np.minimum(sdata, 100);
                sdata = np.maximum(sdata, 0);
                sdata = nibabel.Nifti1Image(sdata, affine=ipaffine)
                input_seg = nibabel.processing.smooth_image(sdata, fwhm)
                input_seg = input_seg.get_fdata()
                input_seg = 0.01*np.asarray(input_seg)
                input_seg = np.where(input_seg < 0.5,  0, input_seg)
                input_seg = np.where(input_seg >= 0.5, 1, input_seg)
                input_seg = nibabel.Nifti1Image(input_seg, affine=ipaffine, header=ipheader)
                            
            # Apply resampling
            resampled_seg0 = nibabel.processing.resample_to_output(input_seg, voxel_sizes=voxel_shape, order=order2)
            opaffine = resampled_seg0.affine
            resampled_seg_data = resampled_seg0.get_fdata()
            
            # This is in case a non-default mask interpolation was chosen
            # and makes sure the mask remains binary
            resampled_seg = np.asarray(resampled_seg_data)
            resampled_seg = np.where(resampled_seg < 0.5,  0, resampled_seg)
            resampled_seg = np.where(resampled_seg >= 0.5, 1, resampled_seg)
            resampled_seg = nibabel.Nifti1Image(resampled_seg, opaffine, header=resampled_seg0.header)
            
            # Attempt to copy sform and qform info from image
            if copyX:
                resampled_seg.set_qform(qformaffine, code=qformcode)
                resampled_seg.set_sform(sformaffine, code=sformcode)
             
            # Ensure mask is saved in sensible format
            data = np.round(resampled_seg.get_fdata()).astype(np.uint8)                                               
            resampled_seg = nibabel.Nifti1Image(data, header=resampled_seg.header, affine=resampled_seg.affine)        
            resampled_seg.header.set_data_dtype(np.uint8)                    
                
            # Save segmentation
            nibabel.save(resampled_seg, outputfile)
                        
    return


def main():
        
    parser = argparse.ArgumentParser("Voxel resampling")

    parser.add_argument("nifti_dir", help="path to nifti directories")
    parser.add_argument("output_dir", help="path to preprocessed output directory")
    parser.add_argument("-vs","--voxel_size", help="target voxel size as 3 dimensions list - average used if non-specified")
    parser.add_argument("-ip", "--image_interpolator", help="image interpolation method", choices=["neighbours","trilinear","cubic"], default="cubic")
    parser.add_argument("-mp", "--mask_interpolator", help="mask interpolation method", choices=["neighbours","trilinear","cubic"], default="neighbours")
    args = parser.parse_args()

    nifti_dir = args.nifti_dir
    output_dir = args.output_dir
    ip = args.image_interpolator
    mp = args.mask_interpolator
    
    if args.voxel_size:
        voxel_size = args.voxel_size

    else:
        voxel_size = average_voxel_calculator(nifti_dir)

    print(f"Average Voxel Size to be used: {voxel_size}")

    Resample2BC(nifti_dir, output_dir, method1=ip, method2=mp, voxel=voxel_size)

if __name__ == "__main__":
    main()