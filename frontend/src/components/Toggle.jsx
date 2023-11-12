
import React, { useRef, useEffect, useState } from "react";



function Toggle({ toggled, onClick }) {
  return (
      <div onClick={onClick} className={`toggle${toggled ? " night" : ""}`}>
          <div className="notch">
              <div className="crater" />
              <div className="crater" />
          </div>
      </div>
  );
}

 
export default Toggle;