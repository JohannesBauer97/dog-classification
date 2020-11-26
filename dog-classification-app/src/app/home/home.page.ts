import {Component, ElementRef, ViewChild} from '@angular/core';
import {CameraResultType, CameraSource, Plugins} from '@capacitor/core';


import * as tf from '@tensorflow/tfjs';
import {GraphModel} from '@tensorflow/tfjs';
import {BreedPrediction} from './breed-prediction';


@Component({
    selector: 'app-home',
    templateUrl: 'home.page.html',
    styleUrls: ['home.page.scss'],
})
export class HomePage {
    @ViewChild('predictImg') predictImg: ElementRef;
    imgSrc: string;
    model: GraphModel;
    isReady = false;
    breedPrediction: BreedPrediction;
    breedlist: string[] = [
        'Chihuahua',
        'Japanese_spaniel',
        'Maltese_dog',
        'Pekinese',
        'Shih-Tzu',
        'Blenheim_spaniel',
        'papillon',
        'toy_terrier',
        'Rhodesian_ridgeback',
        'Afghan_hound',
        'basset',
        'beagle',
        'bloodhound',
        'bluetick',
        'black-and-tan_coonhound',
        'Walker_hound',
        'English_foxhound',
        'redbone',
        'borzoi',
        'Irish_wolfhound',
        'Italian_greyhound',
        'whippet',
        'Ibizan_hound',
        'Norwegian_elkhound',
        'otterhound',
        'Saluki',
        'Scottish_deerhound',
        'Weimaraner',
        'Staffordshire_bullterrier',
        'American_Staffordshire_terrier',
        'Bedlington_terrier',
        'Border_terrier',
        'Kerry_blue_terrier',
        'Irish_terrier',
        'Norfolk_terrier',
        'Norwich_terrier',
        'Yorkshire_terrier',
        'wire-haired_fox_terrier',
        'Lakeland_terrier',
        'Sealyham_terrier',
        'Airedale',
        'cairn',
        'Australian_terrier',
        'Dandie_Dinmont',
        'Boston_bull',
        'miniature_schnauzer',
        'giant_schnauzer',
        'standard_schnauzer',
        'Scotch_terrier',
        'Tibetan_terrier',
        'silky_terrier',
        'soft-coated_wheaten_terrier',
        'West_Highland_white_terrier',
        'Lhasa',
        'flat-coated_retriever',
        'curly-coated_retriever',
        'golden_retriever',
        'Labrador_retriever',
        'Chesapeake_Bay_retriever',
        'German_short-haired_pointer',
        'vizsla',
        'English_setter',
        'Irish_setter',
        'Gordon_setter',
        'Brittany_spaniel',
        'clumber',
        'English_springer',
        'Welsh_springer_spaniel',
        'cocker_spaniel',
        'Sussex_spaniel',
        'Irish_water_spaniel',
        'kuvasz',
        'schipperke',
        'groenendael',
        'malinois',
        'briard',
        'kelpie',
        'komondor',
        'Old_English_sheepdog',
        'Shetland_sheepdog',
        'collie',
        'Border_collie',
        'Bouvier_des_Flandres',
        'Rottweiler',
        'German_shepherd',
        'Doberman',
        'miniature_pinscher',
        'Greater_Swiss_Mountain_dog',
        'Bernese_mountain_dog',
        'Appenzeller',
        'EntleBucher',
        'boxer',
        'bull_mastiff',
        'Tibetan_mastiff',
        'French_bulldog',
        'Great_Dane',
        'Saint_Bernard',
        'Eskimo_dog',
        'malamute',
        'Siberian_husky',
        'affenpinscher',
        'basenji',
        'pug',
        'Leonberg',
        'Newfoundland',
        'Great_Pyrenees',
        'Samoyed',
        'Pomeranian',
        'chow',
        'keeshond',
        'Brabancon_griffon',
        'Pembroke',
        'Cardigan',
        'toy_poodle',
        'miniature_poodle',
        'standard_poodle',
        'Mexican_hairless',
        'dingo',
        'dhole',
        'African_hunting_dog',
    ];

    constructor() {
        this.imgSrc = '/assets/test.jpg';
        this.loadModel();
    }

    async takePhoto(): Promise<void> {

        const capturedPhoto = await Plugins.Camera.getPhoto({
          resultType: CameraResultType.Base64,
          source: CameraSource.Camera,
          quality: 100
        });

        this.imgSrc = 'data:image/png;base64,' + capturedPhoto.base64String;


        const tensor = tf.browser.fromPixels(this.predictImg.nativeElement).expandDims(0).toFloat();
        const prediction = this.model.predict(tensor);
        // console.log('tensor', tensor, 'prediction', prediction);
        // @ts-ignore
        const score = tf.softmax(prediction);
        // console.log('score', score);
        const data = await score.data();
        // console.log('data', data);
        const all: Array<[string, number]> = [];
        for (let i = 0; i < this.breedlist.length; i++) {
            all.push([this.breedlist[i], data[i]]);
        }
        all.sort((a, b) => a[1] - b[1]).reverse();
        const result: BreedPrediction = {
            firstPlace: all[0],
            secondPlace: all[1],
            thirdPlace: all[2],
            all
        };
        this.breedPrediction = result;
    }

    async loadModel() {
        this.model = await tf.loadGraphModel('/assets/modeljs/model.json');
        this.isReady = true;
    }
}
