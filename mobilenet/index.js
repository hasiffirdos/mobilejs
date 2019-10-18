
import * as tf from '@tensorflow/tfjs';
import yolo from "./index1.js";


var THRESHOLD = 0.5;
const Model_path ='http://localhost:1234/tfjs_model_both/model.json';
let myYolo;

const btn = document.getElementById('toggleBtn');
const current_model = document.getElementById('current_model');
btn.innerText = "Snacks";
current_model.innerText = "Selected: Combined";
var class_names = ["Bottle_cap","snacks_BM"];
var Label = "";
var classes = 2;
btn.addEventListener("click", ()=>{
  if (btn.innerText === "Snacks")
  {
    btn.innerText = "Bottle cap";
    current_model.innerText = "Selected: Snacks";
    document.getElementById('file-container').style.display = 'none';
    load_model('http://localhost:1234/tfjs_model_snacks/model.json');
    class_names = ["snacks_BM"];
    Label = "";
    classes = 1;
    

  }
  else if (btn.innerText === "Bottle cap"){
    btn.innerText = "Combined";
    current_model.innerText = "Selected: Bottle Cap";
    document.getElementById('file-container').style.display = 'none';
    load_model('http://localhost:1234/tfjs_model_files/model.json');
    class_names = ["bottle_cap"];
    Label = "";
    classes = 1;

  }else if (btn.innerText === "Combined"){
    btn.innerText = "Snacks";
    current_model.innerText = "Selected: Combined";
    document.getElementById('file-container').style.display = 'none';
    load_model('http://localhost:1234/tfjs_model_both/model.json');
    class_names = ["Bottle_cap","snacks_BM"];
    Label = "";
    classes = 2;

  }


});


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
  ctx.fillText(Label, left, top);
}
function loadImage(canvas_id, img_id){
    var c = document.getElementById(canvas_id);
    var ctx = c.getContext("2d");
    const catElement = document.getElementById(img_id);
    var img_Width = catElement.width;
    var img_Height = catElement.height;
    ctx.canvas.width = img_Width;
    ctx.canvas.height = img_Height;
    ctx.drawImage(catElement,0,0,img_Width,img_Height);
    return c;
}

const load_model = async (c_model_path) => {
  myYolo = await yolo.v3tiny(c_model_path);
  status("model_loaded")
  document.getElementById('file-container').style.display = '';
}

const do_predict = async () => {

  const canvas  = loadImage("my_canvas","cap");

  status('Loading model...');
  // if (myYolo==null){
  //   myYolo = await yolo.v3tiny(Model_path);
  // } else{
  //   console.log("loaded....");

  // }
  
  status("detecting...")
  const boxes = await myYolo.predict(
    canvas,
  {
    maxBoxes: 5,          // defaults to 20
    scoreThreshold: .2,   // defaults to .5
    iouThreshold: .5,     // defaults to .3
    numClasses: classes,       // defaults to 80 for yolo v3, tiny yolo v2, v3 and 20 for tiny yolo v1
    anchors: [4.7439693,3.56052632,  5.96004796,4.44855441,  6.93922585,5.17026503,  7.84043062,5.85085526,  8.85860809,6.59607474,  10.25595052,7.63565918],       // See ./src/config.js for examples   10,14, 23,27, 37,58, 81,82, 135,169, 344,319
    classNames: class_names,    // defaults to coco classes for yolo v3, tiny yolo v2, v3 and voc classes for tiny yolo v1
    inputSize: 480,       // defaults to 416
  });
  if (boxes.length===0 || boxes[0]["score"]<THRESHOLD){
    status("Not detect anything.")
    return 0;
  }
  var out_result = "";
  for(var i = 0; i<boxes.length;i++){
    var boxes_array = [boxes[i]["left"],boxes[i]["right"],boxes[i]["top"],boxes[i]["bottom"]];
    Label = boxes[i]["class"];
    drawBoundingBoxes(canvas,boxes_array);
    out_result+=("Class: ".concat(boxes[0]["class"],"\nScore: ",boxes[0]["score"])+'\n');
  }
  
  status(out_result);


  document.getElementById('file-container').style.display = '';
  return 0;
};


const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.getElementById('cap');
      img.src = e.target.result;
      img.width = 480;
      img.height = 480;
      img.onload = () =>{
        console.time("time_taken");
        do_predict();
        console.timeEnd("time_taken");
        
      }
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');

// do_predict();

(async () => {
  console.log("calling my funcccc******");
  myYolo = await yolo.v3tiny(Model_path);
  document.getElementById('file-container').style.display = '';
})();

//do_predict1();

