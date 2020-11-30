import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {IonicModule} from '@ionic/angular';
import {FormsModule} from '@angular/forms';
import {HomePage} from './home.page';

import {HomePageRoutingModule} from './home-routing.module';
import {FlexLayoutModule} from '@angular/flex-layout';
import {TutorialComponent} from './tutorial/tutorial.component';


@NgModule({
    imports: [
        CommonModule,
        FormsModule,
        IonicModule,
        HomePageRoutingModule,
        FlexLayoutModule
    ],
    declarations: [HomePage, TutorialComponent],
    providers: []
})
export class HomePageModule {
}
