import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import nibabel as nib
from nibabel.processing import conform

from dipy.align import affine_registration, syn_registration, read_mapping, write_mapping
from dipy.align.imaffine import AffineMap


def loadLabels(invol, tlabels):

    # load the image
    labs = nib.load(invol)

    # get the unique labels
    labels = np.unique(labs.get_fdata())[1:]

    # sanity check of text labels vs volume labels
    if all(tlabels["label"].isin(labels)):
        print(" -- Label names match the volume labels.")
    else:
        print(" -- Text label names do not match the volume label names.")
        raise ValueError("Text labels do not match the volume labels.")

    # pull the names from the text file
    roi_labels = tlabels["name"].to_list()

    # sort roi_labels by label number
    roi_labels = [x for _, x in sorted(zip(tlabels["label"], roi_labels))]

    return(labs, labels, roi_labels)


def tryLoad(invol, labs):
    try:

        dat = nib.load(invol)
        vol = dat.get_fdata()
        out = vol
        # print(f" --  --  -- Successfully loaded: {invol}")

    except:

        out = np.empty(labs.shape)
        out[:] = np.nan
        # print(f" --  --  -- Failed to load: {invol}")

    return out


def nanAvg(inp):

    if inp.size == 0:
        out = np.nan

    else:

        try:
            out = np.nanmean(inp)
        except:
            out = np.nan

    return(out)


def qcOverlaySlices(ref, mov, labels=False, slice_index=None, plot_title=None, fname=None, **fig_kwargs):
    """
    Plot three overlaid slices from the given volumes.

    Creates a figure containing three images: the gray scale k-th slice of
    the first volume (ref) to the left, where k=slice_index, the k-th slice of
    the second volume (mov) to the right and the k-th slices of the two given
    images on top of each other using the red channel for the first volume and
    the green channel for the second one. It is assumed that both volumes have
    the same shape. The intended use of this function is to visually assess the
    quality of a registration result.

    Modified from dipy's regtools.overlay_slices. Thanks!

    Parameters
    ----------
    ref : array or NIfTI, shape (S, R, C)
        the reference image from registration plotted to the left.
    mov : array or NIfTI, shape (S, R, C)
        the moving image from the registration plotted to the right.
    labels: bool, optional
        if True, treat the moving image as labels and set the value of all
        non-zero voxels to one for a mask of labeled brain.
    slice_index : int, list, or tuple; optional
        the index of the slices to be overlaid. A single value or list /
        tuple of 3 values can be passed for sagittal, coronal, and axial
        indices. If None, the slice along the specified axis is used.
    plot_title : string, optional
        the title of the plot. If None (default), a generic title is used.
    fname : string, optional
        the name of the file to write the image to. If None (default), the
        figure is not saved to disk.
    fig_kwargs: extra parameters for saving figure, e.g. `dpi=300`.
    """

    # if images are nifti objects, extract the data
    if isinstance(ref, np.ndarray):
        pass
    elif isinstance(ref, nib.Nifti1Image):
        ref = ref.get_fdata()
    elif isinstance(ref, nib.Nifti2Image):
        ref = ref.get_fdata()
    else:
        raise ValueError("ref must be a Nifti1Image, Nifti2Image, or numpy array.")

    if isinstance(mov, np.ndarray):
        pass
    elif isinstance(mov, nib.Nifti1Image):
        mov = mov.get_fdata()
    elif isinstance(mov, nib.Nifti2Image):
        mov = mov.get_fdata()
    else:
        raise ValueError("mov must be a Nifti1Image, Nifti2Image, or numpy array.")

    # Normalize the intensities to [0,255]
    sh = ref.shape
    ref = np.asarray(ref, dtype=np.float64)
    mov = np.asarray(mov, dtype=np.float64)
    ref = 255 * (ref - ref.min()) / (ref.max() - ref.min())

    # for labels, binarize the moving image
    if labels:
        mov = np.where(mov > 0, 255, 0)
        rtitle="Label Mask"
    else:
        mov = 255 * (mov - mov.min()) / (mov.max() - mov.min())
        rtitle="Subject"

    # if slice_index is a single value, assign to all views, otherwise unpack
    if slice_index is None:
        idxs = sh[0] // 2
        idxc = sh[1] // 2
        idxa = sh[2] // 2
    elif isinstance(slice_index, int):
        idxs = idxc = idxa = slice_index
    elif isinstance(slice_index, (list, tuple)) and len(slice_index) == 3:
        idxs, idxc, idxa = slice_index
    else:
        raise ValueError("slice_index must be None, an int, or a list/tuple of 3 ints.")

    # Create the color image to draw the overlapped slices into, and extract
    # the slices (note the transpositions)

    # sagittal
    cs = np.zeros(shape=(sh[2], sh[1], 3), dtype=np.uint8)
    ls = np.asarray(ref[idxs, :, :]).astype(np.uint8).T
    rs = np.asarray(mov[idxs, :, :]).astype(np.uint8).T

    # coronal
    cc = np.zeros(shape=(sh[2], sh[0], 3), dtype=np.uint8)
    lc = np.asarray(ref[:, idxc, :]).astype(np.uint8).T
    rc = np.asarray(mov[:, idxc, :]).astype(np.uint8).T

    # axial
    ca = np.zeros(shape=(sh[1], sh[0], 3), dtype=np.uint8)
    la = np.asarray(ref[:, :, idxa]).astype(np.uint8).T
    ra = np.asarray(mov[:, :, idxa]).astype(np.uint8).T

    # Draw the intensity images to the appropriate channels of the color image
    # The "(ll > ll[0, 0])" condition is just an attempt to eliminate the
    # background when its intensity is not exactly zero (the [0,0] corner is
    # usually background)
    cs[..., 0] = ls * (ls > ls[0, 0])
    cc[..., 0] = lc * (lc > lc[0, 0])
    ca[..., 0] = la * (la > la[0, 0])
    cs[..., 1] = rs * (rs > rs[0, 0])
    cc[..., 1] = rc * (rc > rc[0, 0])
    ca[..., 1] = ra * (ra > ra[0, 0])

    # create a big figure with 3x3 subplots
    fig, ax = plt.subplots(3, 3)
    ax[0, 0].set_axis_off()
    ax[0, 0].imshow(ls, cmap=plt.cm.gray, origin="lower")
    ax[0, 0].set_title("Reference")
    ax[0, 1].set_axis_off()
    ax[0, 1].imshow(cs, cmap=plt.cm.gray, origin="lower")
    ax[0, 1].set_title("Overlay")
    ax[0, 2].set_axis_off()
    ax[0, 2].imshow(rs, cmap=plt.cm.gray, origin="lower")
    ax[0, 2].set_title(rtitle)
    ax[1, 0].set_axis_off()
    ax[1, 0].imshow(lc, cmap=plt.cm.gray, origin="lower")
    ax[1, 1].set_axis_off()
    ax[1, 1].imshow(cc, cmap=plt.cm.gray, origin="lower")
    ax[1, 2].set_axis_off()
    ax[1, 2].imshow(rc, cmap=plt.cm.gray, origin="lower")
    ax[2, 0].set_axis_off()
    ax[2, 0].imshow(la, cmap=plt.cm.gray, origin="lower")
    ax[2, 1].set_axis_off()
    ax[2, 1].imshow(ca, cmap=plt.cm.gray, origin="lower")
    ax[2, 2].set_axis_off()
    ax[2, 2].imshow(ra, cmap=plt.cm.gray, origin="lower")

    if plot_title is None:
        plot_title = "Registration QC Overlay Slices"
    fig.suptitle(plot_title, fontsize=16)

    # set figure dpi (and size kinda)
    fig.set_dpi(300)

    # Save the figure to disk, if requested
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight", **fig_kwargs)

    return fig


parser = argparse.ArgumentParser(description="load data by subject and create dataframes of stats to merge and plot.")
parser.add_argument("-d", "--derivatives", action='append', help='Derivatives directory(s) to crawl for IDP features', required=True)
parser.add_argument("-p", "--subject", help="subject ID(s) to selectively extract", action='append')
parser.add_argument("-s", "--session", help="participant session to extract")
parser.add_argument("-r", "--label_ref", help="label volume to reference during alignment estimation")
parser.add_argument("-v", "--label_vol", help="label volume in reference space to extract ROI data")
parser.add_argument("-l", "--label_lab", help="label volume labels that match volume data")
parser.add_argument("-a", "--affine", help="file stem of affine to look for to shortcut alignment", default=None)
parser.add_argument("-f", "--force", nargs="?", const=1, type=bool, default=False, help="re-export all found subjects")
args = parser.parse_args()

pipelines = args.derivatives
sids = args.subject
sess = args.session
label_ref = args.label_ref
label_vol = args.label_vol
label_lab = args.label_lab
label_aff = args.affine
redo = args.force

# pipelines = ["/home/bcmcpher/Projects/nipoppy/qpn-subset-0.4.0/derivatives/dmri_freewater_qsiprep", "/home/bcmcpher/Projects/nipoppy/qpn-subset-0.4.0/derivatives/dmri_freewater_tractoflow"]
# subj = "sub-40533"
# sess = "BL"
# label_ref = "/home/bcmcpher/Projects/nipoppy/qpn-subset-0.4.0/code/labels/JHU-FA.nii.gz"
# label_vol = "/home/bcmcpher/Projects/nipoppy/qpn-subset-0.4.0/code/labels/JHU-tracts.nii.gz"
# label_lab = "/home/bcmcpher/Projects/nipoppy/qpn-subset-0.4.0/code/labels/JHU-tracts.tsv"
# label_aff = "JHU"

print("Running extraction of estimated features.")

#
# load label volume
#

print("Loading and verifying label volume.")

# load the label names to verify
tlabels = pd.read_csv(label_lab, sep="\t")

# check and load the label image
labs, labels, roi_labels = loadLabels(label_vol, tlabels)
parc = Path(label_vol).name.replace(".nii.gz", "")
nlabs = len(roi_labels)

# get the labels data - will be transformed to subject space
tldat = labs.get_fdata()

# load the reference volume for coregistration
ref = nib.load(label_ref)

# the path to the shared affine transform
saffdir = Path(Path(label_vol).parent.absolute(), "affine")

#
# figure out pipeline paths to crawl and write to
#

# for each input pipeline
for pipe in pipelines:

    # get top level label for IDPs
    # pname = os.path.basename(pipe).replace("dmri_freewater_", "")
    # pvers = os.listdir(Path(pipe))[0]
    # print(f"Pipeline - Version: {pname}-{pvers}")
    pname = os.path.basename(pipe)
    pvers = "2.0.0"
    print(f"Pipeline - Version: {pname}-{pvers}")

    # build input / output paths
    datadir = Path(pipe, pvers, "output")
    outsdir = Path(pipe, pvers, "idps")

    # generator of subject folders
    subjs = os.listdir(Path(pipe, pvers, "output"))
    print(f" -- N subjects found: {len(list(subjs))}")

    # if sids given, filter to those
    if sids is not None:
        subjs = [s for s in subjs if s in sids]
        print(f" -- N subjects after filtering: {len(list(subjs))}")

    # for every subject
    print(f" -- Extracting IDP data from: {pname}-{pvers}")
    for subj in subjs:

        # create sub ID w/o prefix
        sub = subj.replace("sub-", "")

        # create output file name
        outfile = Path(outsdir, f"sub-{sub}_ses-{sess}_{pname}-{pvers}_{parc}_idps.tsv")

        # if the file exists and no redo flag, skip iteration
        if outfile.exists() & (not redo):
            print(f" --  -- Data already extracted for: {subj}. Skipping.")
            continue

        # load the files to extract
        print(f" --  -- Extracting data from: {subj}")
        dpdir = Path(datadir, subj, f"ses-{sess}", "dipy")
        spdir = Path(datadir, subj, f"ses-{sess}", "scilpy")

        # loading bvals to check the kind of data
        try:
            bval = np.loadtxt(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_desc-fwcorr_dwi.bval"))
            if len(np.unique(bval)) < 3:
                print(" --  --  -- Single-shell data loaded")
                tshell = "single-shell"
            else:
                print(" --  --  -- Multi-shell data loaded")
                tshell = "multi-shell"
        except:
            print(" --  --  -- Bval not found?")
            tshell = "missing"

        #
        # linear + nonlinear align dipy DTI FA (dpdtfa) to input ref
        #

        # path to subject affine file
        subj_aff_stem = f"sub-{sub}_ses-{sess}_{pname}-{pvers}_{label_aff}_sub2mni_affine.txt"
        subj_wrp_stem = f"sub-{sub}_ses-{sess}_{pname}-{pvers}_{label_aff}_sub2mni_warp.nii.gz"
        subj_aff = Path(saffdir, subj_aff_stem)
        subj_wrp = Path(saffdir, subj_aff_stem)
        subj_qcd = Path(saffdir, "qc")

        # if the affine file exists, load it
        if subj_aff.exists() & subj_wrp.exists():

            print(f" --  --  -- Using existing affine: {subj_aff_stem}")

            # load the text affine file
            sub2mni_aff = np.loadtxt(subj_aff)
            mni2sub_aff = np.linalg.inv(sub2mni_aff)

            # load the subject space image
            mov = nib.load(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-fa_map.nii.gz"))

            # load the warp field - affine must exist or it should be redone
            sub2mni_wrp = read_mapping("sub2mni-warp.nii.gz", mov, ref, prealign=mni2sub_aff)

        # otherwise compute alignment of FA to template
        else:
            print(" --  --  -- Alignment files not found. Computing.")

            try:
                mov = nib.load(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-fa_map.nii.gz"))
            except:
                print(" -- -- Failed to load DT FA image for alignment. Nothing can be done.")
                continue

            # perform affine registration
            _, sub2mni_aff = affine_registration(
                mov,
                ref,
                moving_affine=mov.affine,
                static_affine=ref.affine,
                nbins=32,
                metric="MI",
                pipeline=["center_of_mass", "translation", "rigid", "affine"],
                level_iters=[10000, 1000, 100],
                sigmas=[3.0, 1.0, 0.0],
                factors=[4, 2, 1],
            )

            # perform nonlinear registration
            _, sub2mni_wrp = syn_registration(
                mov,
                ref,
                moving_affine=mov.affine,
                static_affine=ref.affine,
                prealign=sub2mni_aff)

        # save the affine to disk
        if not subj_aff.exists():
            saffdir.mkdir(parents=True, exist_ok=True)
            print(f" --  --  -- Saving affine to: {subj_aff}")
            np.savetxt(subj_aff, sub2mni_aff)

        # save the warp to disk
        if not subj_wrp.exists():
            print(f" --  --  -- Saving warp to: {subj_wrp}")
            write_mapping(sub2mni_wrp, subj_wrp)

        # get the invervse affine
        mni2sub_aff = np.linalg.inv(sub2mni_aff)

        # create the affine map object to apply linear transform
        affmap = AffineMap(
            sub2mni_aff,
            domain_grid_shape=ref.shape,
            domain_grid2world=ref.affine,
            codomain_grid_shape=mov.shape,
            codomain_grid2world=mov.affine)

        # transform the moving image w/ linear affine only for qc
        qc1 = affmap.transform(mov.get_fdata())

        # create a QC image of the linear alignment
        qcOverlaySlices(ref, qc1,
                        plot_title=f"sub-{sub}_ses-{sess} {pname} Linear Alignment QC",
                        fname=Path(subj_qcd, f"sub-{sub}_ses-{sess}_{pname}_{ref}_linear_qc.png"))

        # create the inverse warp object to go from MNI to subject space
        qc2 = sub2mni_wrp.transform_inverse(mov.get_fdata())

        # create a QC image of the nonlinear alignment
        qcOverlaySlices(ref, qc2,
                        plot_title=f"sub-{sub}_ses-{sess} {pname} Nonlinear Alignment QC",
                        fname=Path(subj_qcd, f"sub-{sub}_ses-{sess}_{pname}_{ref}_warp_qc.png"))

        # apply the nonliear warp
        tlabs = sub2mni_wrp.transform_inverse(tldat,
                                              interpolation="nearest",
                                              out_shape=mov.shape,
                                              out_grid2world=mov.affine)

        # create a QC image of the labels over subject alignment
        qcOverlaySlices(ref, tlabs, labels=True,
                        plot_title=f"sub-{sub}_ses-{sess} {pname} Warped Labels QC",
                        fname=Path(subj_qcd, f"sub-{sub}_ses-{sess}_{pname}_{ref}_labels_qc.png"))

        #
        # load and prep data for extraction
        #

        # load data files
        print(" --  --  -- Loading data files for extraction.")
        dpdtfa = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-fa_map.nii.gz"), tlabs)
        dpdtmd = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-md_map.nii.gz"), tlabs)
        dpdtrd = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-rd_map.nii.gz"), tlabs)
        dpdtad = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-ad_map.nii.gz"), tlabs)
        dpdtse = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-nrmse_map.nii.gz"), tlabs)
        dpdtrs = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-residual_map.nii.gz"), tlabs)

        dpfwfa = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-fa_map.nii.gz"), tlabs)
        dpfwmd = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-md_map.nii.gz"), tlabs)
        dpfwrd = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-rd_map.nii.gz"), tlabs)
        dpfwad = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-ad_map.nii.gz"), tlabs)
        dpfwfw = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-freewater_map.nii.gz"), tlabs)
        dpfwse = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-nrmse_map.nii.gz"), tlabs)
        dpfwrs = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-residual_map.nii.gz"), tlabs)

        spfwfa = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-fa_map.nii.gz"), tlabs)
        spfwmd = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-md_map.nii.gz"), tlabs)
        spfwrd = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-rd_map.nii.gz"), tlabs)
        spfwad = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-ad_map.nii.gz"), tlabs)
        spfwfw = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-freewater_map.nii.gz"), tlabs)
        spfwse = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-nrmse_map.nii.gz"), tlabs)
        spfwrs = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-residual_map.nii.gz"), tlabs)

        # preallocate output lists
        dpdtfa_mean = []
        dpdtmd_mean = []
        dpdtrd_mean = []
        dpdtad_mean = []
        dpdtfw_mean = np.zeros(nlabs)
        dpdtse_mean = []
        dpdtrs_mean = []

        dpfwfa_mean = []
        dpfwmd_mean = []
        dpfwrd_mean = []
        dpfwad_mean = []
        dpfwfw_mean = []
        dpfwse_mean = []
        dpfwrs_mean = []

        spfwfa_mean = []
        spfwmd_mean = []
        spfwrd_mean = []
        spfwad_mean = []
        spfwfw_mean = []
        spfwse_mean = []
        spfwrs_mean = []

        # for every roi label, get the mean value w/in the labels
        for idx, roi in enumerate(labels):
            dpdtfa_mean.append(nanAvg(dpdtfa[tldat == roi]))
            dpdtmd_mean.append(nanAvg(dpdtmd[tldat == roi]))
            dpdtrd_mean.append(nanAvg(dpdtrd[tldat == roi]))
            dpdtad_mean.append(nanAvg(dpdtad[tldat == roi]))
            dpdtse_mean.append(nanAvg(dpdtse[tldat == roi]))
            dpdtrs_mean.append(nanAvg(dpdtrs[tldat == roi]))

            dpfwfa_mean.append(nanAvg(dpfwfa[tldat == roi]))
            dpfwmd_mean.append(nanAvg(dpfwmd[tldat == roi]))
            dpfwrd_mean.append(nanAvg(dpfwrd[tldat == roi]))
            dpfwad_mean.append(nanAvg(dpfwad[tldat == roi]))
            dpfwfw_mean.append(nanAvg(dpfwfw[tldat == roi]))
            dpfwse_mean.append(nanAvg(dpfwse[tldat == roi]))
            dpfwrs_mean.append(nanAvg(dpfwrs[tldat == roi]))

            spfwfa_mean.append(nanAvg(spfwfa[tldat == roi]))
            spfwmd_mean.append(nanAvg(spfwmd[tldat == roi]))
            spfwrd_mean.append(nanAvg(spfwrd[tldat == roi]))
            spfwad_mean.append(nanAvg(spfwad[tldat == roi]))
            spfwfw_mean.append(nanAvg(spfwfw[tldat == roi]))
            spfwse_mean.append(nanAvg(spfwse[tldat == roi]))
            spfwrs_mean.append(nanAvg(spfwrs[tldat == roi]))

        #
        # create output dataframes
        #

        # merge regular dipy tensor
        dpdt_data = pd.DataFrame([roi_labels,
                                  dpdtfa_mean,
                                  dpdtmd_mean,
                                  dpdtrd_mean,
                                  dpdtad_mean,
                                  dpdtfw_mean,
                                  dpdtse_mean,
                                  dpdtrs_mean])
        dpdt_data = dpdt_data.T

        dpdt_data["subj"] = sub
        dpdt_data["pipe"] = f"{pname}-{pvers}"
        dpdt_data["soft"] = "dipy"
        dpdt_data["shell"] = tshell
        dpdt_data["model"] = "dti"

        # label and reorder columns
        dpdt_data.columns = ["roi_labels", "fa", "md", "rd", "ad", "fw", "nrmse", "residual", "subj", "pipeline", "software", "shell", "model"]
        dpdt_data = dpdt_data[["subj", "pipeline", "software", "shell", "model", "roi_labels", "fa", "md", "rd", "ad", "fw", "nrmse", "residual"]]

        # merge fw dipy tensor
        dpfw_data = pd.DataFrame([roi_labels,
                                  dpfwfa_mean,
                                  dpfwmd_mean,
                                  dpfwrd_mean,
                                  dpfwad_mean,
                                  dpfwfw_mean,
                                  dpfwse_mean,
                                  dpfwrs_mean])
        dpfw_data = dpfw_data.T

        dpfw_data["subj"] = sub
        dpfw_data["pipe"] = f"{pname}-{pvers}"
        dpfw_data["soft"] = "dipy"
        dpfw_data["shell"] = tshell
        dpfw_data["model"] = "fwdti"

        # label and reorder columns
        dpfw_data.columns = ["roi_labels", "fa", "md", "rd", "ad", "fw", "nrmse", "residual", "subj", "pipeline", "software", "shell", "model"]
        dpfw_data = dpfw_data[["subj", "pipeline", "software", "shell", "model", "roi_labels", "fa", "md", "rd", "ad", "fw", "nrmse", "residual"]]

        # merge scilpy fw tensor
        spfw_data = pd.DataFrame([roi_labels,
                                  spfwfa_mean,
                                  spfwmd_mean,
                                  spfwrd_mean,
                                  spfwad_mean,
                                  spfwfw_mean,
                                  spfwse_mean,
                                  spfwrs_mean])
        spfw_data = spfw_data.T

        spfw_data["subj"] = sub
        spfw_data["pipe"] = f"{pname}-{pvers}"
        spfw_data["soft"] = "scilpy"
        spfw_data["shell"] = tshell
        spfw_data["model"] = "fwdti"

        # label and reorder columns
        spfw_data.columns = ["roi_labels", "fa", "md", "rd", "ad", "fw", "nrmse", "residual", "subj", "pipeline", "software", "shell", "model"]
        spfw_data = spfw_data[["subj", "pipeline", "software", "shell", "model", "roi_labels", "fa", "md", "rd", "ad", "fw", "nrmse", "residual"]]

        # merge the dataframes
        sdata = pd.concat([dpdt_data, dpfw_data, spfw_data])

        # write the dataframe to disk
        sdata.to_csv(outfile, index=False, header=True, sep="\t")
        print(f" --  -- Saved sub-{sub}_ses-{sess} IDPs to disk.")

print("Done.")
