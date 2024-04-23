const video = document.getElementById("video");

async function getCameraStream() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    video.play();
  } catch (err) {
    console.error("Error accessing camera:", err);
  }
}

function captureFrame() {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.8);
}

async function sendFrameToServer() {
  const frameData = captureFrame();
  const response = await fetch("/process_frame", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ frame: frameData }),
  });

  const processedFrame = await response.json();
  if (processedFrame.status === "success") {
    const processedImage = document.getElementById("processed-image");
    processedImage.src = processedFrame.processed_frame;
  } else {
    console.error("Error processing frame:", processedFrame.message);
  }
}

getCameraStream();
setInterval(sendFrameToServer, 1000); // 每隔1000毫秒（1秒）发送一次
