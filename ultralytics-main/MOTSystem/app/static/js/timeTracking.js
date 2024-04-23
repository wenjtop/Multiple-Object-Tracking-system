const timeTracking = document.getElementById("video");

async function getCameraStream() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    timeTracking.srcObject = stream;
    timeTracking.play();
  } catch (err) {
    console.error("Error accessing camera:", err);

  }
}

function captureFrame() {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  ctx.drawImage(timeTracking, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.8);
}
function getCookie(name) {
  const value = "; " + document.cookie;
  const parts = value.split("; " + name + "=");
  if (parts.length === 2) return parts.pop().split(";").shift();
}

const queue = [];
var preTime = Number(new Date().getTime())
async function sendFrameToServer() {
  const frameData = captureFrame();
  const csrfToken = getCookie("csrftoken");
  queue.push(Number(new Date().getTime()));
  const response = await fetch("/timeTracking/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrfToken,
    },
    body: JSON.stringify({ frame: frameData , time: new Date().getTime()}),
  });
  const processedFrame = await response.json();
  queue.shift();
  if (processedFrame.status === "success" && Number(processedFrame.time)>preTime) {
    preTime = Number(processedFrame.time)
    var count = processedFrame.count
    const processedcount = document.getElementById("count");
    processedcount.innerText ="计数：" + count
    const processedImage = document.getElementById("processed-image");
    processedImage.src = processedFrame.processed_frame;
  } else {
    console.error("Error processing frame:", processedFrame.message);
  }
}

getCameraStream();

async function sendCont() {
  if (Number(new Date().getTime()) - queue[0] > 1000){
    queue.shift();
  }
  if (queue.length < 5){
    sendFrameToServer();
  }
}
setInterval(sendCont, 1000/50); // 每隔1000毫秒（1秒）发送一次
