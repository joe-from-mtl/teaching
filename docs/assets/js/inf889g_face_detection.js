// Load the trained detector
// From: https://github.com/shantnu/Webcam-Face-Detect/blob/master/haarcascade_frontalface_default.xml
// And: https://docs.opencv.org/3.4/df/d6c/tutorial_js_face_detection_camera.html

let utils = new Utils('errorMessage');
let streaming = false;
let videoInput = document.getElementById('videoInput');
let startAndStop = document.getElementById('startAndStop');
let canvasOutput = document.getElementById('canvasOutput');
let canvasContext = canvasOutput.getContext('2d');
let faceCascadeFile = "haarcascade_frontalface_default.xml"
let videoSelect = document.querySelector('select#videoSource')
var selectors = [videoSelect];

function gotDevices(deviceInfos) {
    // Handles being called several times to update labels. Preserve values.
    var values = selectors.map(function(select) {
        return select.value;
    });
    selectors.forEach(function(select) {
        while (select.firstChild) {
            select.removeChild(select.firstChild);
        }
    });
    for (var i = 0; i !== deviceInfos.length; ++i) {
        var deviceInfo = deviceInfos[i];
        var option = document.createElement('option');
        option.value = deviceInfo.deviceId;
        if (deviceInfo.kind === 'videoinput') {
            option.text = 'Cam' + (videoSelect.length + 1) + ' - ' + deviceInfo.label;
            videoSelect.appendChild(option);
        }
    }
    selectors.forEach(function(select, selectorIndex) {
        if (Array.prototype.slice.call(select.childNodes).some(function(n) {
            return n.value === values[selectorIndex];
        })) {
            select.value = values[selectorIndex];
        }
    });
}

function handleError(error) {
    console.log('navigator.getUserMedia error: ', error);
}

navigator.mediaDevices.enumerateDevices().then(gotDevices).catch(handleError);


function gotStream(stream) {
    window.stream = stream; // make stream available to console
    videoInput.srcObject = stream;
    // Refresh button list in case labels have become available
    return navigator.mediaDevices.enumerateDevices();
}

function start() {
    if (window.stream) {
        window.stream.getTracks().forEach(function(track) {
            track.stop();
        });
    }
    var videoSource = videoSelect.value;
    var constraints = {
        audio: false,
        video: {deviceId: videoSource ? {exact: videoSource} : undefined,
            width: {exact: 320}, height: {exact: 240}}
    };
    if (streaming == true) {
        navigator.mediaDevices.getUserMedia(constraints).
        then(gotStream).then(gotDevices).catch(handleError);
    }
}

videoSelect.onchange = start;





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
    load_cascadeFiles();
});

function load_cascadeFiles() {
    utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
        console.log('cascade ready to load.');
        startAndStop.removeAttribute('disabled');
    });
}

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

function main(){
    let classifier = new cv.CascadeClassifier();
    classifier.load(faceCascadeFile);

    let video = document.getElementById('videoInput');
    let cap = new cv.VideoCapture(video);
    let frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let gray = new cv.Mat();
    cap.read(frame);
    cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY, 0);
    let msize = new cv.Size(0, 0);

    let faces = new cv.RectVector;

    function update() {
        try {
            if (!streaming) {
                // clean and stop.
                frame.delete();
                gray.delete();
                faces.delete();
                classifier.delete();
                return;
            }
            // Read a new video frame
            cap.read(frame);

            // Convert to gray
            cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY, 0);

            // Detect faces
            classifier.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);

            // Draw faces.
            for (let i = 0; i < faces.size(); ++i) {
                let face = faces.get(i);
                let point1 = new cv.Point(face.x, face.y);
                let point2 = new cv.Point(face.x + face.width, face.y + face.height);
                cv.rectangle(frame, point1, point2, [255, 0, 0, 255]);
            }

            // Display
            cv.imshow('canvasOutput', frame);

            // schedule the next one.
            requestAnimationFrame(update);
        } catch(err) {
            console.log(err);
            //utils.printError(err);
        }
    }
    update();
}

