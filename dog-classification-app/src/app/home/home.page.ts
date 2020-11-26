import {Component, ElementRef, ViewChild} from '@angular/core';
import { Plugins, CameraResultType, CameraSource } from '@capacitor/core';
const { Camera } = Plugins;
import * as tf from '@tensorflow/tfjs';
import { GraphModel } from '@tensorflow/tfjs';


@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage {
  @ViewChild('predictImg') predictImg: ElementRef;
  img: string;
  model: GraphModel;

  constructor() {
    this.loadModel();
  }

  async takePhoto(): Promise<void>{
    /*
    const capturedPhoto = await Camera.getPhoto({
      resultType: CameraResultType.Base64,
      source: CameraSource.Camera,
      quality: 100
    });
     */
    console.log(this.model.inputNodes);
    const imageUrl = '/assets/test.jpg';
    console.log(this.predictImg.nativeElement);
    const tensors = tf.browser.fromPixels(this.predictImg.nativeElement);
    console.log('tensors', tensors);
    console.log('tensor dtype fix', tensors.toFloat());
    const prediction = this.model.predict(tensors.toFloat().expandDims(0));
    console.log('prediction', prediction);
    // @ts-ignore
    const score = tf.softmax(prediction);
    console.log('score', score);
  }

  async loadModel(){
    this.model = await tf.loadGraphModel('/assets/modeljs/model.json');
  }
}
