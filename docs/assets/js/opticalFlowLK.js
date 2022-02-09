// Create the GUI
let gui = new dat.GUI();
gui.closed = true;
gui.hide();

let parameters_corner = {maxCorners: 30, qualityLevel: 0.3, minDistance: 7, blockSize: 7}
let parameters_lk = {winSize: 15, maxLevel: 2,}

let parameters = {
    pyrScale: 2, levels: 3, winsize: 15, iterations: 3,
    polyN: 5, polySigma: 1.2, useGaussian: false, overlay: false
}

let corner_gui = gui.addFolder("ShiTomasi Corner Detection")
corner_gui.add(parameters_corner, "maxCorners", 1, 100);
corner_gui.add(parameters_corner, "qualityLevel", 0, 1, 0.1);
corner_gui.add(parameters_corner, "minDistance", 1, 15, 1);
corner_gui.add(parameters_corner, "blockSize", 1, 15, 1);

let lk_gui = gui.addFolder("LK Optical Flow")
lk_gui.add(parameters_lk, "winSize", 1, 21, 1);
lk_gui.add(parameters_lk, "maxLevel", 1, 6, 1);


function main() {
    let video = document.getElementById('videoInput');
    let cap = new cv.VideoCapture(video);

    // parameters for lucas kanade optical flow
    let winSize = new cv.Size(parameters_lk.winSize, parameters_lk.winSize);
    let criteria = new cv.TermCriteria(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03);

    // create some random colors
    let color = [];
    for (let i = 0; i < parameters_corner.maxCorners; i++) {
        color.push(new cv.Scalar(parseInt(Math.random() * 255), parseInt(Math.random() * 255),
            parseInt(Math.random() * 255), 255));
    }

    // take first frame and find corners in it
    let oldFrame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    cap.read(oldFrame);
    let oldGray = new cv.Mat();
    cv.cvtColor(oldFrame, oldGray, cv.COLOR_RGB2GRAY);
    let p0 = new cv.Mat();
    let none = new cv.Mat();
    cv.goodFeaturesToTrack(oldGray, p0, parameters_corner.maxCorners, parameters_corner.qualityLevel, parameters_corner.minDistance, none, parameters_corner.blockSize);

    // Create a mask image for drawing purposes
    let zeroEle = new cv.Scalar(0, 0, 0, 255);
    let mask = new cv.Mat(oldFrame.rows, oldFrame.cols, oldFrame.type(), zeroEle);

    let frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let frameGray = new cv.Mat();
    let p1 = new cv.Mat();
    let st = new cv.Mat();
    let err = new cv.Mat();

    //const FPS = 30;

    function processVideo() {
        try {
            if (!streaming) {
                // clean and stop.
                frame.delete();
                oldGray.delete();
                p0.delete();
                p1.delete();
                err.delete();
                mask.delete();
                return;
            }
            //let begin = Date.now();

            // start processing.
            cap.read(frame);
            cv.cvtColor(frame, frameGray, cv.COLOR_RGBA2GRAY);

            // calculate optical flow
            winSize = new cv.Size(parameters_lk.winSize, parameters_lk.winSize);
            cv.calcOpticalFlowPyrLK(oldGray, frameGray, p0, p1, st, err, winSize, parameters_lk.maxLevel, criteria);

            // select good points
            let goodNew = [];
            let goodOld = [];
            for (let i = 0; i < st.rows; i++) {
                if (st.data[i] === 1) {
                    goodNew.push(new cv.Point(p1.data32F[i * 2], p1.data32F[i * 2 + 1]));
                    goodOld.push(new cv.Point(p0.data32F[i * 2], p0.data32F[i * 2 + 1]));
                }
            }

            // draw the tracks
            for (let i = 0; i < goodNew.length; i++) {
                cv.line(mask, goodNew[i], goodOld[i], color[i], 2);
                cv.circle(frame, goodNew[i], 5, color[i], -1);
            }
            cv.add(frame, mask, frame);

            cv.imshow('canvasOutput', frame);

            // now update the previous frame and previous points
            frameGray.copyTo(oldGray);
            p0.delete();
            p0 = null;
            p0 = new cv.Mat(goodNew.length, 1, cv.CV_32FC2);
            for (let i = 0; i < goodNew.length; i++) {
                p0.data32F[i * 2] = goodNew[i].x;
                p0.data32F[i * 2 + 1] = goodNew[i].y;
            }

            // schedule the next one.
            requestAnimationFrame(processVideo);
        } catch (err) {
            console.log(err);
        }
    }
    requestAnimationFrame(processVideo);
}

let utils = new Utils('errorMessage');

let streaming = false;
let videoInput = document.getElementById('videoInput');
let startAndStop = document.getElementById('startAndStop');
let canvasOutput = document.getElementById('canvasOutput');
let canvasContext = canvasOutput.getContext('2d');

startAndStop.addEventListener('click', () => {
    if (!streaming) {
        utils.startCamera('qvga', onVideoStarted, 'videoInput');
    } else {
        utils.stopCamera();
        onVideoStopped();
    }
});

function onVideoStarted() {
    streaming = true;
    startAndStop.innerText = 'Stop';
    startAndStop.classList.remove("btn-primary")
    startAndStop.classList.add("btn-danger")
    videoInput.width = videoInput.videoWidth;
    videoInput.height = videoInput.videoHeight;
    main();
}

function onVideoStopped() {
    streaming = false;
    //canvasContext.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
    startAndStop.innerText = 'Start';
    startAndStop.classList.remove("btn-danger")
    startAndStop.classList.add("btn-primary")
}

utils.loadOpenCv(() => {
    startAndStop.removeAttribute('disabled');
});