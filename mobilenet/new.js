import * as tf from '@tensorflow/tfjs';
import yolo from "./index1.js"

const Model_path ='http://localhost:1234/tfjs_model_files/model.json';
let myYolo;

function drawBoundingBoxes(canvas, trueBoundingBox) {
    tf.util.assert(
        trueBoundingBox != null && trueBoundingBox.length === 4,
        `Expected boundingBoxArray to have length 4, ` +
            `but got ${trueBoundingBox} instead`);
  
    let left = trueBoundingBox[0];
    let right = trueBoundingBox[1];
    let top = trueBoundingBox[2];
    let bottom = trueBoundingBox[3];
  
    const ctx = canvas.getContext('2d');
    ctx.beginPath();
    ctx.strokeStyle = 'rgb(255,0,0)';
    ctx.lineWidth = 2;
    ctx.moveTo(left, top);
    ctx.lineTo(right, top);
    ctx.lineTo(right, bottom);
    ctx.lineTo(left, bottom);
    ctx.lineTo(left, top);
    ctx.stroke();
  
    ctx.font = '15px Arial';
    ctx.fillStyle = 'rgb(255,0,0)';
    ctx.fillText('true', left, top);
  }
 
  
function predict(){
// //loading Image in canvas 
//     var c = document.getElementById("my_canvas");
//     var ctx = c.getContext("2d");
//     const catElement = document.getElementById('cat');
//     var img_Width = catElement.width;
//     var img_Height = catElement.height;
//     ctx.canvas.width = img_Width;
//     ctx.canvas.height = img_Height;
//     ctx.drawImage(newImg,0,0,img_Width,img_Height);

    const canvas  = loadImage(canvas_id = "my_canvas",img_id = "cat");

    status('Loading model...');
    myYolo = await yolo.v3tiny(Model_path);
    const boxes = await myYolo.predict(
        canvas,
      {
        maxBoxes: 5,          // defaults to 20
        scoreThreshold: .2,   // defaults to .5
        iouThreshold: .5,     // defaults to .3
        numClasses: 1,       // defaults to 80 for yolo v3, tiny yolo v2, v3 and 20 for tiny yolo v1
        anchors: [4.7439693,3.56052632,  5.96004796,4.44855441,  6.93922585,5.17026503,  7.84043062,5.85085526,  8.85860809,6.59607474,  10.25595052,7.63565918],       // See ./src/config.js for examples   10,14, 23,27, 37,58, 81,82, 135,169, 344,319
        classNames: ["Bottle_cap"],    // defaults to coco classes for yolo v3, tiny yolo v2, v3 and voc classes for tiny yolo v1
        inputSize: 480,       // defaults to 416
      });
      // const predictions = await model.predict(expanded);
      var boxes_array = [boxes[0]["left"],boxes[0]["right"],boxes[0]["top"],boxes[0]["bottom"]]
      drawBoundingBoxes(canvas,boxes_array);
      status(boxes[0]["class"]);
    
    
      document.getElementById('file-container').style.display = '';
      return 0;

}


function loadImage(canvas_id, img_id){
    var c = document.getElementById(canvas_id);
    var ctx = c.getContext("2d");
    const catElement = document.getElementById(img_id);
    var img_Width = catElement.width;
    var img_Height = catElement.height;
    ctx.canvas.width = img_Width;
    ctx.canvas.height = img_Height;
    ctx.drawImage(newImg,0,0,img_Width,img_Height);
    return c;
}