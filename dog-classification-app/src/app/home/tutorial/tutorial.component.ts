import { Component, OnInit } from '@angular/core';
import {ModalController} from '@ionic/angular';

@Component({
  selector: 'app-tutorial',
  templateUrl: './tutorial.component.html',
  styleUrls: ['./tutorial.component.scss'],
})
export class TutorialComponent implements OnInit {

  constructor(private modalController: ModalController) { }

  ngOnInit() {}

  close(): void{
    this.modalController.dismiss();
  }

}
