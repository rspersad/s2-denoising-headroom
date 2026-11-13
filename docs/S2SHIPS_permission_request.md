Subject: Request for permission to redistribute derived S2-SHIPS annotations with a doctoral data release

Dear Dr. Ciocarlan and Dr. Stoian,

I am completing a Doctor of Engineering Praxis at The George Washington University, "Perceptual Quality Versus Detection Performance: Recoverable Headroom as a Basis for Denoising Decisions in Sentinel-2 Vessel Detection," which used S2-SHIPS as its annotation source. Thank you for making the dataset available; it is cited throughout (Ciocarlan & Stoian, 2021, Remote Sensing 13(21), 4255).

To make the results reproducible, I am publishing the analysis code, results tables and processed imagery in a public, read-only repository and archive (GitHub and Zenodo). The imagery itself is Copernicus Sentinel-2 data and is redistributed under the Copernicus licence. The vessel labels, however, derive from your annotations, and S2-SHIPS carries no public license. I would therefore like to ask your permission to include the following derived files:

  1. YOLO-format bounding-box label files for the [N] S2-SHIPS scenes used ([list scene identifiers]), re-tiled to 640 x 640 pixels;
  2. the vessel-level ground-truth table used for scoring (scene, vessel id, box coordinates in pixel units).

I would publish them under CC BY 4.0, with the requirement that users cite your 2021 article, and with a NOTICE file crediting S2-SHIPS. If you would prefer a narrower scope (for example, a non-commercial license, or label files for the four held-out scenes only), I will follow it exactly; and if you would prefer that the annotations not be redistributed, I will instead publish a script that rebuilds the label files from a user's own copy of S2-SHIPS obtained from you.

A reply by email stating the permitted scope is all that is needed; I will keep it on file with the release.

With thanks and best regards,

Rajesh Surendra Persad
Doctor of Engineering candidate, The George Washington University
