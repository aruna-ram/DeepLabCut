#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#


def select_cropping_area(config, videos=None):
    """
    Interactively select the cropping area of all videos in the config.
    A user interface pops up with a frame to select the cropping parameters.
    Use the left click to draw a box and hit the button 'set cropping parameters'
    to store the cropping parameters for a video in the config.yaml file.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : optional (default=None)
        List of videos whose cropping areas are to be defined. Note that full paths are required.
        By default, all videos in the config are successively loaded.

    Returns
    -------
    cfg : dict
        Updated project configuration
    """
    from deeplabcut.utils import auxiliaryfunctions, auxfun_videos

    cfg = auxiliaryfunctions.read_config(config)
    if videos is None:
        videos = list(cfg.get("video_sets_original") or cfg["video_sets"])

    for video in videos:
        coords = auxfun_videos.draw_bbox(video)
        if coords:
            temp = {
                "crop": ", ".join(
                    map(
                        str,
                        [
                            int(coords[0]),
                            int(coords[2]),
                            int(coords[1]),
                            int(coords[3]),
                        ],
                    )
                )
            }
            try:
                cfg["video_sets"][video] = temp
            except KeyError:
                cfg["video_sets_original"][video] = temp

    auxiliaryfunctions.write_config(config, cfg)
    return cfg


def extract_frames(
    config,
    mode="automatic",
    algo="kmeans",
    crop=False,
    userfeedback=True,
    cluster_step=1,
    cluster_resizewidth=30,
    cluster_color=False,
    opencv=True,
    slider_width=25,
    config3d=None,
    extracted_cam=0,
    videos_list=None,
):
    """Extracts frames from the project videos.

    Frames will be extracted from videos listed in the config.yaml file.

    The frames can be selected in several ways:

    - ``automatic``: frames are extracted either randomly and temporally uniformly
    (``uniform``) or by clustering based on visual appearance (``k-means``).
    - ``manual``: user selects frames interactively using a GUI.
    - ``match``: matched frames are extracted from additional cameras for epipolar labeling.
    - ``all``: **all frames from all videos are extracted**. This ignores any frame number
    restrictions in the config file and extracts every frame in the video(s).

    Please refer to the user guide for more details on methods and parameters
    https://www.nature.com/articles/s41596-019-0176-0 or the preprint:
    https://www.biorxiv.org/content/biorxiv/early/2018/11/24/476531.full.pdf

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    mode : string. Either ``"automatic"``, ``"manual"``, ``"match"``, or ``"all"``.
        String containing the mode of extraction.
        - ``automatic`` or ``manual`` extracts the initial set of frames.
        - ``match`` extracts frames from additional cameras to match previously extracted frames.
        - ``all`` extracts **every frame** from all videos, ignoring ``numframes2pick`` in the config.

    algo : string, Either ``"kmeans"`` or ``"uniform"``, Default: `"kmeans"`.
        Only used in ``automatic`` mode. Specifies the algorithm for selecting frames.

    crop : bool or str, optional
        If ``True``, video frames are cropped according to the coordinates in the config.
        If cropping coordinates are unknown, ``crop="GUI"`` opens a GUI to define them.

    userfeedback: bool, optional
        If ``False`` in ``automatic`` mode, frames for all videos are extracted automatically.
        If ``True``, the user is asked for each video whether to extract frames.

    cluster_resizewidth: int, default: 30
        For ``k-means`` clustering, the width to which images are downsampled.

    cluster_step: int, default: 1
        Step size for frame sampling before clustering (saves memory).

    cluster_color: bool, default: False
        If ``True``, color channels are considered in k-means clustering.

    opencv: bool, default: True
        Use OpenCV for loading/extraction (otherwise MoviePy is used).

    slider_width: int, default: 25
        Width of the video frames slider (manual mode).

    config3d: string, optional
        Path to the 3D project config for ``match`` mode.

    extracted_cam: int, default: 0
        Camera index for ``match`` mode.

    videos_list: list[str], Default: None
        Subset of videos to extract frames from. Defaults to all videos in config.

    Returns
    -------
    None

    Notes
    -----
    - ``all`` mode overrides ``numframes2pick`` and extracts every frame.
    - Cropping is still applied if ``crop=True`` or ``"GUI"``.
    - Use ``add_new_videos`` to add new videos to the config and extract frames.

    Examples
    --------
    # Extract all frames from all videos
    >>> deeplabcut.extract_frames(config='/analysis/project/reaching-task/config.yaml', mode='all')

    # Original automatic extraction with k-means
    >>> deeplabcut.extract_frames(config='/analysis/project/reaching-task/config.yaml',
                                mode='automatic', algo='kmeans', crop=True)
    """
    import os
    import sys
    import re
    import glob
    import numpy as np
    from pathlib import Path
    from skimage import io
    from skimage.util import img_as_ubyte
    from deeplabcut.utils import frameselectiontools
    from deeplabcut.utils import auxiliaryfunctions

    config_file = Path(config).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)
    print("Config file read successfully.")


    if opencv:
        from deeplabcut.utils.auxfun_videos import VideoWriter
    else:
        from moviepy.editor import VideoFileClip

    if videos_list is None:
        videos = list(cfg.get("video_sets_original") or cfg["video_sets"])
    else:  # filter video_list by the ones in the config file
        videos = [v for v in cfg["video_sets"] if v in videos_list]

    if mode == "manual":
        from deeplabcut.gui.widgets import launch_napari

        _ = launch_napari(videos[0])
        return

    elif mode == "automatic":
        numframes2pick = cfg["numframes2pick"]
        start = cfg["start"]
        stop = cfg["stop"]

        # Check for variable correctness
        if start > 1 or stop > 1 or start < 0 or stop < 0 or start >= stop:
            raise Exception(
                "Erroneous start or stop values. Please correct it in the config file."
            )
        if numframes2pick < 1 and not int(numframes2pick):
            raise Exception(
                "Perhaps consider extracting more, or a natural number of frames."
            )

        ##====INSERT BY ARUNA==============
    elif mode == "all":
        # New mode: extract ALL frames
        if opencv:
            cap = VideoWriter(video)
            nframes = len(cap)
        else:
            clip = VideoFileClip(video)
            nframes = int(np.ceil(clip.duration * clip.fps))
        #TODO:insert cap? not sure what 

        frames2pick = range(nframes)
        print(f"Extracting ALL {nframes} frames from video: {video}")

        has_failed = []
        for video in videos:
            if userfeedback:
                print(
                    "Do you want to extract (perhaps additional) frames for video:",
                    video,
                    "?",
                )
                askuser = input("yes/no")
            else:
                askuser = "yes"

            if (
                askuser == "y"
                or askuser == "yes"
                or askuser == "Ja"
                or askuser == "ha"
                or askuser == "oui"
                or askuser == "ouais"
            ):  # multilanguage support :)
                if opencv:
                    cap = VideoWriter(video)
                    nframes = len(cap)
                else:
                    # Moviepy:
                    clip = VideoFileClip(video)
                    fps = clip.fps
                    nframes = int(np.ceil(clip.duration * 1.0 / fps))
                if not nframes:
                    print("Video could not be opened. Skipping...")
                    continue

                indexlength = int(np.ceil(np.log10(nframes)))

                fname = Path(video)
                output_path = Path(config).parents[0] / "labeled-data" / fname.stem

                if output_path.exists():
                    if len(os.listdir(output_path)):
                        if userfeedback:
                            askuser = input(
                                "The directory already contains some frames. Do you want to add to it?(yes/no): "
                            )
                        if not (
                            askuser == "y"
                            or askuser == "yes"
                            or askuser == "Y"
                            or askuser == "Yes"
                        ):
                            sys.exit("Delete the frames and try again later!")

                if crop == "GUI":
                    cfg = select_cropping_area(config, [video])
                try:
                    coords = cfg["video_sets"][video]["crop"].split(",")
                except KeyError:
                    coords = cfg["video_sets_original"][video]["crop"].split(",")

                if crop:
                    if opencv:
                        cap.set_bbox(*map(int, coords))
                    else:
                        clip = clip.crop(
                            y1=int(coords[2]),
                            y2=int(coords[3]),
                            x1=int(coords[0]),
                            x2=int(coords[1]),
                        )
                else:
                    coords = None

                print("Extracting frames based on %s ..." % algo)
                if algo == "uniform":
                    if opencv:
                        frames2pick = frameselectiontools.UniformFramescv2(
                            cap, numframes2pick, start, stop
                        )
                    else:
                        frames2pick = frameselectiontools.UniformFrames(
                            clip, numframes2pick, start, stop
                        )
                elif algo == "kmeans":
                    if opencv:
                        frames2pick = frameselectiontools.KmeansbasedFrameselectioncv2(
                            cap,
                            numframes2pick,
                            start,
                            stop,
                            step=cluster_step,
                            resizewidth=cluster_resizewidth,
                            color=cluster_color,
                        )
                    else:
                        frames2pick = frameselectiontools.KmeansbasedFrameselection(
                            clip,
                            numframes2pick,
                            start,
                            stop,
                            step=cluster_step,
                            resizewidth=cluster_resizewidth,
                            color=cluster_color,
                        )
                else:
                    print(
                         "Please implement this method yourself and send us a pull "
                         "request! Otherwise, choose 'uniform' or 'kmeans'."
                     )
                    frames2pick = []

                if not len(frames2pick):
                    print("Frame selection failed...")
                    return []

                output_path = (
                    Path(config).parents[0] / "labeled-data" / Path(video).stem
                )
                output_path.mkdir(parents=True, exist_ok=True)
                is_valid = []
                if opencv:
                    for index in frames2pick:
                        cap.set_to_frame(index)  # extract a particular frame
                        frame = cap.read_frame(crop=True)
                        if frame is not None:
                            image = img_as_ubyte(frame)
                            img_name = (
                                str(output_path)
                                + "/img"
                                + str(index).zfill(indexlength)
                                + ".png"
                            )
                            io.imsave(img_name, image)
                            is_valid.append(True)
                        else:
                            print("Frame", index, " not found!")
                            is_valid.append(False)
                    cap.close()
                else:
                    for index in frames2pick:
                        try:
                            image = img_as_ubyte(clip.get_frame(index * 1.0 / clip.fps))
                            img_name = (
                                str(output_path)
                                + "/img"
                                + str(index).zfill(indexlength)
                                + ".png"
                            )
                            io.imsave(img_name, image)
                            if np.var(image) == 0:  # constant image
                                print(
                                    "Seems like black/constant images are extracted from your video. Perhaps consider using opencv under the hood, by setting: opencv=True"
                                )
                            is_valid.append(True)
                        except FileNotFoundError:
                            print("Frame # ", index, " does not exist.")
                            is_valid.append(False)
                    clip.close()
                    del clip

                if not any(is_valid):
                    has_failed.append(True)
                else:
                    has_failed.append(False)

            else:  # NO!
                has_failed.append(False)

        if all(has_failed):
            print("Frame extraction failed. Video files must be corrupted.")
            return has_failed
        elif any(has_failed):
            print("Although most frames were extracted, some were invalid.")
        else:
            print(
                "Frames were successfully extracted, for the videos listed in the config.yaml file."
            )
        print(
            "\nYou can now label the frames using the function 'label_frames' "
            "(Note, you should label frames extracted from diverse videos (and many videos; we do not recommend training on single videos!))."
        )
        return has_failed

    elif mode == "match":
        import cv2

        config_file = Path(config).resolve()
        cfg = auxiliaryfunctions.read_config(config_file)
        print("Config file read successfully.")
        videos = sorted(cfg["video_sets"].keys())
        if videos_list is not None:  # filter video_list by the ones in the config file
            videos = [v for v in videos if v in videos_list]
        project_path = Path(config).parents[0]
        labels_path = os.path.join(project_path, "labeled-data/")
        video_dir = os.path.join(project_path, "videos/")
        try:
            cfg_3d = auxiliaryfunctions.read_config(config3d)
        except:
            raise Exception(
                "You must create a 3D project and edit the 3D config file before extracting matched frames. \n"
            )
        cams = cfg_3d["camera_names"]
        extCam_name = cams[extracted_cam]
        del cams[extracted_cam]
        label_dirs = sorted(
            glob.glob(os.path.join(labels_path, "*" + extCam_name + "*"))
        )

        # select crop method
        crop_list = []
        for video in videos:
            if extCam_name in video:
                if crop == "GUI":
                    cfg = select_cropping_area(config, [video])
                    print("in gui code")
                coords = cfg["video_sets"][video]["crop"].split(",")

                if crop and not opencv:
                    clip = clip.crop(
                        y1=int(coords[2]),
                        y2=int(coords[3]),
                        x1=int(coords[0]),
                        x2=int(coords[1]),
                    )
                elif not crop:
                    coords = None
                crop_list.append(coords)

        for coords, dirPath in zip(crop_list, label_dirs):
            extracted_images = glob.glob(os.path.join(dirPath, "*png"))

            imgPattern = re.compile("[0-9]{1,10}")
            for cam in cams:
                output_path = re.sub(extCam_name, cam, dirPath)

                for fname in os.listdir(output_path):
                    if fname.endswith(".png"):
                        os.remove(os.path.join(output_path, fname))

                # Find the matching video from the config `video_sets`,
                # as it may be stored elsewhere than in the `videos` directory.
                video_name = os.path.basename(output_path)
                vid = ""
                for video in cfg["video_sets"]:
                    if video_name in video:
                        vid = video
                        break
                if not vid:
                    raise ValueError(f"Video {video_name} not found...")

                cap = cv2.VideoCapture(vid)
                print("\n extracting matched frames from " + video_name)
                for img in extracted_images:
                    imgNum = re.findall(imgPattern, os.path.basename(img))[0]
                    cap.set(1, int(imgNum))
                    ret, frame = cap.read()
                    if ret:
                        image = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        img_name = os.path.join(output_path, "img" + imgNum + ".png")
                        if crop:
                            io.imsave(
                                img_name,
                                image[
                                    int(coords[2]) : int(coords[3]),
                                    int(coords[0]) : int(coords[1]),
                                    :,
                                ],
                            )
                        else:
                            io.imsave(img_name, image)
        print(
            "\n Done extracting matched frames. You can now begin labeling frames using the function label_frames\n"
        )

    else:
        print(
            "Invalid MODE. Choose either 'manual', 'automatic' or 'match'. Check ``help(deeplabcut.extract_frames)`` on python and ``deeplabcut.extract_frames?`` \
              for ipython/jupyter notebook for more details."
        )
