# Is there a way to build a fast, small app quickly for lepinet inference

Goal: prediction app using the camera or the photo galery of a phone or desktop
and giving a species, genus and family names of a lepidoptera photo. 


Additional goals: The app also gives model confidence and highlight the levels
above a minimum confidence threshold, the other levels are greyed. If no family
is above threshold, then the app question the presence of a lepidopter or state
that the displayed species may not be part of the training dataset. For the
lowel level above threshold, the app display three pictures within its training
database. For each level, the app also gives the hyperlink toward the GBIF page.

The app should be light and fast and run offline.

## User interface

The app could look like a Google translate app. 

When launched, you get a home page with text explaining what this app is and in
the middle, a big camera button that opens the camera of the user and a small
galery button on its left side that open the phone galery. This could also be
just a camera button and when the camera opens, there is a galery botton on the
side. Ideally, rely on the default camera app, so the app does not have to
include camera management such as rotation, zoom etc. At the bottom of this home
page a link redirect a user who would like to learn more or give feedback to the
GitHub page.

After a picture has been taken, it shrinks down at the top tier of the screen
and the prediction appears almost instantenously at the bottom of the screen. A
loading symbol could be necessery in the future while waiting for the
prediction, but it would be great to not need it. 

In the middle tier, the predictions family, genus and species names are
presented on top of each others. There names can be copied to be searched. The
user can either highlight the text or a little copy symbol can be used. A
confidence score is displayed next to each level with visual cues. For instance,
this confidence number between 00 and 99 could be included inside a progress bar
circling around it that goes from 00 to 99 and from red to green. The taxa name
is greyed if the level of confidence is below a certain threshold. This
threshold could be 0.5, but this may change in the future. Next to each taxa
name, a hyperlink symbol redirect toward GBIF page associated with this taxa.
This 3x3 informations should be cleaning displayed in a structured fashion.

In the bottom tier of the display, three sample pictures of the lowest
highlighted level are displayed. They can be clicked to be displayed bigger. A
text below inform about the taxa name.

The user can then use the return button of their phone to go back to the home
page.

## Roadmap

This section describes the roadmap of the development process of this app.

The app should be as small as possible and run locally on the user phone or
desktop. 

### Shrinking the model down to the small possible size

This section tackles the difficult pro

### Creating the app

### Future avenues of development

- Storing capture images locally or having some sort of identification history.

## Development rules