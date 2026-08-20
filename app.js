const API_BASE = "http://127.0.0.1:8001";

const FEATURES = [
  "Temperature","Humidity","WindSpeed","GeneralDiffuseFlows","DiffuseFlows",
  "lag_1","lag_6","lag_144","lag_1008","TotalConsumption"
];

const DEMO = {
  Temperature: 25,
  Humidity: 70,
  WindSpeed: 2,
  GeneralDiffuseFlows: 150,
  DiffuseFlows: 80,
  lag_1: 32000,
  lag_6: 31500,
  lag_144: 31000,
  lag_1008: 30000,
  TotalConsumption: 70000
};

function buildTable(){
  const body=document.getElementById("sequenceBody");
  body.innerHTML="";
  for(let r=0;r<10;r++){
    const tr=document.createElement("tr");
    const step=document.createElement("td");
    step.textContent=r+1;
    tr.appendChild(step);
    FEATURES.forEach(f=>{
      const td=document.createElement("td");
      const input=document.createElement("input");
      input.type="number"; input.step="any";
      input.dataset.feature=f; input.dataset.row=r;
      td.appendChild(input); tr.appendChild(td);
    });
    body.appendChild(tr);
  }
}

function baseObservation(){
  const row={};
  for(const f of FEATURES){
    const input=document.getElementById(f);
    const v=Number(input.value);
    if(!Number.isFinite(v)) throw new Error(`Enter a valid value for ${f}.`);
    row[f]=v;
  }
  return row;
}

function copyToAll(){
  const row=baseObservation();
  document.querySelectorAll("#sequenceBody input").forEach(el=>{
    el.value=row[el.dataset.feature];
  });
  document.getElementById("message").textContent="Observation copied to all 10 time steps.";
}

function loadDemo(){
  for(const f of FEATURES) document.getElementById(f).value=DEMO[f];
  copyToAll();
}

function clearAll(){
  for(const f of FEATURES) document.getElementById(f).value="";
  document.querySelectorAll("#sequenceBody input").forEach(el=>el.value="");
  document.getElementById("prediction").textContent="—";
  document.getElementById("demandStatus").textContent="Waiting";
  document.getElementById("response").textContent="{}";
  document.getElementById("message").textContent="";
}

function readSequence(){
  return Array.from(document.querySelectorAll("#sequenceBody tr")).map(tr=>{
    const obj={};
    tr.querySelectorAll("input").forEach(el=>{
      const v=Number(el.value);
      if(!Number.isFinite(v)) throw new Error(`Complete step ${Number(el.dataset.row)+1}.`);
      obj[el.dataset.feature]=v;
    });
    return obj;
  });
}

function statusFor(value){
  const el=document.getElementById("demandStatus");
  if(value<25000){el.textContent="Low";el.style.color="#34d399";}
  else if(value<40000){el.textContent="Normal";el.style.color="#fbbf24";}
  else{el.textContent="High";el.style.color="#fb7185";}
}

async function checkAPI(){
  try{
    const r=await fetch(`${API_BASE}/health`);
    const d=await r.json();
    if(r.ok && d.status==="healthy"){
      const s=document.getElementById("apiStatus");
      s.textContent="API Online"; s.className="status online";
    }
  }catch(e){
    const s=document.getElementById("apiStatus");
    s.textContent="API Offline"; s.className="status offline";
  }
}

async function predict(){
  const btn=document.getElementById("predictBtn");
  const msg=document.getElementById("message");
  btn.disabled=true; msg.textContent="Sending 10 observations to Flask...";

  try{
    const observations=readSequence();
    const r=await fetch(`${API_BASE}/predict`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({observations})
    });
    const data=await r.json();
    document.getElementById("response").textContent=JSON.stringify(data,null,2);
    if(!r.ok || !data.success) throw new Error(data.error || "Prediction failed.");

    const value=Number(data.predicted_consumption);
    document.getElementById("prediction").textContent=value.toLocaleString(undefined,{maximumFractionDigits:2});
    document.getElementById("unit").textContent=data.unit || "consumption units";
    statusFor(value);
    msg.textContent="Prediction completed successfully.";
    msg.style.color="#34d399";
  }catch(e){
    msg.textContent=e.message;
    msg.style.color="#fb7185";
  }finally{
    btn.disabled=false;
  }
}

document.addEventListener("DOMContentLoaded",()=>{
  buildTable();
  checkAPI();
  document.getElementById("demoBtn").addEventListener("click",loadDemo);
  document.getElementById("copyBtn").addEventListener("click",()=>{try{copyToAll()}catch(e){document.getElementById("message").textContent=e.message}});
  document.getElementById("clearBtn").addEventListener("click",clearAll);
  document.getElementById("predictBtn").addEventListener("click",predict);
});
