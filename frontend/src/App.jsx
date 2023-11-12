import { useEffect, useState } from 'react'
import React from 'react';
import './App.css'
import Face from './Face';
import { FileInput, Checkbox, Label } from 'flowbite-react';
import axios from 'axios';
import { PictureContext } from './context/context';



const URL= 'http://127.0.0.1:5000'
function App() {
  const [check, setCheck] = useState(false)
  const [picture, setPicture] = useState([])
  const [image, setImage] = useState([])

  const URL= 'http://127.0.0.1:5000'

  const base64ImagesArray = [];

const handlePost = (event) => {
  const files = event.target.files;

  for (const file of files) {
    const reader = new FileReader();

    reader.onload = async (e) => {
      const base64Image = e.target.result.split(',')[1];
      const imageName = file.name;
      base64ImagesArray.push({ name: imageName, photo: base64Image });
      setImage(base64ImagesArray)
      if (base64ImagesArray.length === files.length) {
        // Выполните POST-запрос на сервер, отправив массив на сервер
        axios.post(URL + '/api/photo',  base64ImagesArray)
          .then(response => {
            const data = response.data;
            // setPicture(data)
            setImage(data)
            console.log(data)
            axios.post(URL+'/api/download', data, )
            .then(response=>{
              const url = window.URL.createObjectURL(new Blob([response.data]));
              const a = document.createElement('a');
              a.href = url;
              a.download = 'result.csv';//сюда имя
              a.click();
              window.URL.revokeObjectURL(url);
              })
          })
          .catch(error => {
            console.error('Error:', error);
          });
      }
    };

    reader.readAsDataURL(file);
  }
};

 
 
  return (
    <PictureContext.Provider value={{picture, setPicture}}>
<div className='w-[96%] relative  mx-auto p-2 '>
      <div className="flex bg-neutral-600 fixed w-full items-center p-3 left-0 top-0">
      <FileInput
      multiple
        className='w-full mr-5'
          id="file"
          onChange={handlePost}
        />
        <Checkbox
          id="accept"
          onChange={(e)=>setCheck(e.target.checked)}
        />
        <label
          className="flex min-w-max text-white text-2xl ml-2"
          htmlFor="agree"
        >
            Использовать камеру
        
        </label>
        </div>
        <div>
     
        {/* <img className='inline-block object-cover ml-3 rounded-lg h-[82px] w-[82px]' src={picture}/> */}
      </div>
      
      {check
      ?
       <Face/>
      :
        (image.map((pic, index)=>(
          <div className='my-20'>
          <p className='m-5 text-7xl'>{index} {pic.log}</p>
          <img className='inline-block object-cover ml-3 rounded-lg h-[960px] w-[960px]' src={'data:image/jpeg;base64,'+pic.photo}/>
          </div>
        
        ))
          
        )
      }
  </div>
    </PictureContext.Provider>
    
  )
}

export default App
