function getCookie(name) {
  const value = "; " + document.cookie;
  const parts = value.split("; " + name + "=");
  if (parts.length === 2) return parts.pop().split(";").shift();
}

const queue = [];
var preTime = 0
var num = 0
var pathname = window.location.pathname;
async function sendFrameToServer() {
  num = num + 1
  queue.push(Number(new Date().getTime()));
  const csrfToken = getCookie("csrftoken");
  if (num>=100){
    window.clearInterval(interval);
  }
  if(num>=24) {
    const response1 = await fetch(pathname, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrfToken,
    },
    body: JSON.stringify({time: new Date().getTime(), "end": 'YES'}),
  });
    const processedFrame1 = await response1.json();
    if (processedFrame1.status === "success" && processedFrame1.end =="YES") {
      window.clearInterval(interval);
    }
  }
  const response = await fetch(pathname, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrfToken,
    },
    body: JSON.stringify({time: new Date().getTime(), "end": 'NO'}),
  });

  const processedFrame = await response.json();
  queue.shift();
  if (processedFrame.status === "success" && preTime < Number(processedFrame.time)) {
    num = 0
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

// getCameraStream();

async function sendCont() {
  if (Number(new Date().getTime()) - queue[0] > 1000){
    queue.shift();
  }
  if (queue.length < 1){
    sendFrameToServer();
  }
  else{
    console.log('no')
  }
}

var interval = setInterval(sendCont, 1000/20); // 每隔1000毫秒（1秒）发送一次

