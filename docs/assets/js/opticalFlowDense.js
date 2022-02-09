// Create the GUI
let gui = new dat.GUI();
gui.closed = true;
gui.hide();

let parameters = {
    pyrScale: 2, levels: 3, winsize: 15, iterations: 3,
    polyN: 5, polySigma: 1.2, useGaussian: false, overlay: false
}
gui.add(parameters, "pyrScale", 2, 5, 1);
gui.add(parameters, "levels", 1, 4, 1);
gui.add(parameters, "winsize", 1, 21, 1);
gui.add(parameters, "iterations", 1, 15, 1);
gui.add(parameters, "polyN", 1, 15, 1);
gui.add(parameters, "polySigma", 0.5, 9, 0.1);
gui.add(parameters, "useGaussian");
gui.add(parameters, "overlay");

// Main function
function main() {
    let video = document.getElementById('videoInput');
    let cap = new cv.VideoCapture(video);

    // take first frame of the video
    let frame1 = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    cap.read(frame1);

    let prvs = new cv.Mat();
    cv.cvtColor(frame1, prvs, cv.COLOR_RGBA2GRAY);
    frame1.delete();
    let hsv = new cv.Mat();
    let hsv0 = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let hsv1 = new cv.Mat(video.height, video.width, cv.CV_8UC1, new cv.Scalar(255));
    let hsv2 = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let hsvVec = new cv.MatVector();
    hsvVec.push_back(hsv0);
    hsvVec.push_back(hsv1);
    hsvVec.push_back(hsv2);

    let frame2 = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let next = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let flow = new cv.Mat(video.height, video.width, cv.CV_32FC2);
    let flowVec = new cv.MatVector();
    let mag = new cv.Mat(video.height, video.width, cv.CV_32FC1);
    let ang = new cv.Mat(video.height, video.width, cv.CV_32FC1);
    let rgb = new cv.Mat(video.height, video.width, cv.CV_8UC3);

    //const FPS = 30;

    function processVideo() {
        try {
            if (!streaming) {
                // clean and stop.
                prvs.delete();
                hsv.delete();
                hsv0.delete();
                hsv1.delete();
                hsv2.delete();
                hsvVec.delete();
                frame2.delete();
                flow.delete();
                flowVec.delete();
                next.delete();
                mag.delete();
                ang.delete();
                rgb.delete();
                return;
            }
            //let begin = Date.now();

            // start processing.
            cap.read(frame2);
            cv.cvtColor(frame2, next, cv.COLOR_RGBA2GRAY);
            let flags = 0
            if (parameters.useGaussian) {
                flags += cv.OPTFLOW_FARNEBACK_GAUSSIAN;
            }
            cv.calcOpticalFlowFarneback(prvs, next, flow, 1 / parameters.pyrScale,
                parameters.levels, parameters.winsize, parameters.iterations,
                parameters.polyN, parameters.polySigma, flags);
            cv.split(flow, flowVec);
            let u = flowVec.get(0);
            let v = flowVec.get(1);
            cv.cartToPolar(u, v, mag, ang);
            u.delete();
            v.delete();
            ang.convertTo(hsv0, cv.CV_8UC1, 180 / Math.PI / 2);
            if (parameters.overlay) {
                next.copyTo(hsv2);
            } else {
                cv.normalize(mag, hsv2, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1);
            }
            cv.merge(hsvVec, hsv);
            cv.cvtColor(hsv, rgb, cv.COLOR_HSV2RGB);
            cv.imshow('canvasOutput', rgb);
            next.copyTo(prvs);

            // schedule the next one.
            //let delay = 1000/FPS - (Date.now() - begin);
            //setTimeout(processVideo, delay);
            requestAnimationFrame(processVideo);
        } catch (err) {
            //utils.printError(err);
            console.log(err);
        }
    }

    // schedule the first one.
    setTimeout(processVideo, 0);
}


let utils = new Utils('errorMessage');

//utils.loadCode('codeSnippet', 'codeEditor');

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
