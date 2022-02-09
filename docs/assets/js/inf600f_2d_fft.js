// https://docs.opencv.org/3.4/dd/d02/tutorial_js_fourier_transform.html

function main() {
    const FPS = 30;
    let video = document.getElementById('videoInput');
    let cap = new cv.VideoCapture(video);

    // take first frame of the video
    let frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let src = new cv.Mat();
    cap.read(frame);
    cv.cvtColor(frame, src, cv.COLOR_RGBA2GRAY, 0);

    function processVideo() {
        try {
            if (!streaming) {
                // clean and stop.
                src.delete();
                return;
            }

            // start processing.
            let begin = Date.now();
            cap.read(frame);
            cv.cvtColor(frame, src, cv.COLOR_RGBA2GRAY, 0);

            // get optimal size of DFT
            let optimalRows = cv.getOptimalDFTSize(src.rows);
            let optimalCols = cv.getOptimalDFTSize(src.cols);
            let s0 = cv.Scalar.all(0);
            let padded = new cv.Mat();
            cv.copyMakeBorder(src, padded, 0, optimalRows - src.rows, 0,
                optimalCols - src.cols, cv.BORDER_CONSTANT, s0);

            // use cv.MatVector to distribute space for real part and imaginary part
            let plane0 = new cv.Mat();
            padded.convertTo(plane0, cv.CV_32F);
            let planes = new cv.MatVector();
            let complexI = new cv.Mat();
            let plane1 = new cv.Mat.zeros(padded.rows, padded.cols, cv.CV_32F);
            planes.push_back(plane0);
            planes.push_back(plane1);
            cv.merge(planes, complexI);

            // in-place dft transform
            cv.dft(complexI, complexI);

            // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
            cv.split(complexI, planes);
            cv.magnitude(planes.get(0), planes.get(1), planes.get(0));
            let mag = planes.get(0);
            let m1 = new cv.Mat.ones(mag.rows, mag.cols, mag.type());
            cv.add(mag, m1, mag);
            cv.log(mag, mag);

            // crop the spectrum, if it has an odd number of rows or columns
            let rect = new cv.Rect(0, 0, mag.cols & -2, mag.rows & -2);
            mag = mag.roi(rect);

            // rearrange the quadrants of Fourier image
            // so that the origin is at the image center
            let cx = mag.cols / 2;
            let cy = mag.rows / 2;
            let tmp = new cv.Mat();

            let rect0 = new cv.Rect(0, 0, cx, cy);
            let rect1 = new cv.Rect(cx, 0, cx, cy);
            let rect2 = new cv.Rect(0, cy, cx, cy);
            let rect3 = new cv.Rect(cx, cy, cx, cy);

            let q0 = mag.roi(rect0);
            let q1 = mag.roi(rect1);
            let q2 = mag.roi(rect2);
            let q3 = mag.roi(rect3);

            // exchange 1 and 4 quadrants
            q0.copyTo(tmp);
            q3.copyTo(q0);
            tmp.copyTo(q3);

            // exchange 2 and 3 quadrants
            q1.copyTo(tmp);
            q2.copyTo(q1);
            tmp.copyTo(q2);

            // The pixel value of cv.CV_32S type image ranges from 0 to 1.
            cv.normalize(mag, mag, 0, 1, cv.NORM_MINMAX);

            cv.imshow('canvasOutput', mag);
            padded.delete(); planes.delete(); complexI.delete(); m1.delete(); tmp.delete();

            // schedule the next one.
            let delay = 1000/FPS - (Date.now() - begin);
            setTimeout(processVideo, delay);
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