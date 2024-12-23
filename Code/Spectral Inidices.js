//Sentinel2



Map.addLayer(ROI1)
var SA=ROI1.first();
var SA_Fe=ee.Feature(SA);
var ROI=SA_Fe.buffer(50).bounds().geometry(); 
Map.addLayer(ROI)

function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

function NDVI(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}

function GSAVI(image){
  var gsavi= image.expression(
    '((NIR-G)/(NIR+G+0.5)) * (1+0.5)',
    {'NIR': image.select('B8'),
      'G':image.select('B3'),
    }).rename('GSAVI');
    return image.addBands(gsavi);
}

function GNDVI(image){
  
  var gndvi= image.expression(
    '(NIR-G)/(NIR+G)',
    {'NIR': image.select('B8'),
      'G':image.select('B3'),
    }).rename('GNDVI');
    return image.addBands(gndvi);
  
}

function CVI(image){
  
  var cvi=image.expression(
    '(NIR/G)*(R/G)',
    {'NIR': image.select('B8'),
      'G':image.select('B3'),
      'R':image.select('B4'),
    }).rename('CVI');
    return image.addBands(cvi);
  
}

function NDGI(image){
    var ndgi = image.normalizedDifference(['B3', 'B4']).rename('NDGI');
  return image.addBands(ndgi)
}

function NBR(image){
  var nbr=image.normalizedDifference(['B8', 'B12']).rename('NBR');
  return image.addBands(nbr)
}

function NDII(image){
  var ndii= image.normalizedDifference(['B8', 'B11']).rename('NDII');
  return image.addBands(ndii)
}

function GDVI(image){
  var gdvii=image.expression(
    'NIR-G',
    {'NIR': image.select('B8'),
      'G':image.select('B3'),
    }).rename('GDVII');
    return image.addBands(gdvii)
}

function MSAVI(image){
  var msavi=image.expression(
    '(2 * NIR + 1 - sqrt( ((2 * NIR + 1)**2) - (8 * (NIR - R)) ))/2' , 
    {'NIR': image.select('B8'),
      'R':image.select('B4'),}).rename('MSAVI');
      return image.addBands(msavi);
}

function DVI(image){
  var dvi=image.expression(
    'NIR-R',
    {'NIR': image.select('B8'),
      'R':image.select('B4'),
    }).rename('DVI');
    return image.addBands(dvi);
}

function SAVI(image){
  var savi=image.expression(
    '((NIR-R)/(NIR+R+0.5)) * (1+0.5)',
    {'NIR': image.select('B8'),
      'R':image.select('B4'),
    }).rename('SAVI');
    return image.addBands(savi);
}

function MSR(image){
  var msr=image.expression(
    '((NIR/R) - 1 ) / ( sqrt((NIR/R) + 1 ) )',
    {'NIR': image.select('B8'),
      'R':image.select('B4'),
    }).rename('MSR');
    return image.addBands(msr);
}

function SI(image){
  var rg=image.expression('R/G',{'R':image.select('B4'), 'G': image.select('B3')}).rename('RG');
  var s1n=image.expression('SWIR1/NIR',{'SWIR1':image.select('B11'), 'NIR': image.select('B8')}).rename('S1N');
  var ng=image.expression('NIR/G',{'G':image.select('B3'), 'NIR': image.select('B8')}).rename('NG');
  var s2g=image.expression('SWIR2/G',{'G':image.select('B3'), 'SWIR2': image.select('B12')}).rename('S2G');
  var nr=image.expression('NIR/R',{'R':image.select('B4'), 'NIR': image.select('B8')}).rename('NR');
  var s2r=image.expression('SWIR2/R',{'SWIR2':image.select('B12'), 'R': image.select('B4')}).rename('S2R');
  var s1g=image.expression('SWIR1/G',{'SWIR1':image.select('B11'), 'G': image.select('B3')}).rename('S1G');
  var s2n=image.expression('SWIR2/NIR',{'SWIR2':image.select('B12'), 'NIR': image.select('B8')}).rename('S2N');
  var s1r=image.expression('SWIR1/R',{'SWIR1':image.select('B11'), 'R': image.select('B4')}).rename('S1R');
  var swir21=image.expression('SWIR2/SWIR1',{'SWIR2':image.select('B12'), 'SWIR1': image.select('B11')}).rename('SWIR21');
  
  return image.addBands([rg,s1n,ng,s2g,nr,s2r,s1g,s2n,s1r,swir21]);
}
//////2018
//Warm  : '2018-05-20', '2018-09-20'
//Cold :  '2018-12-01', '2019-03-01'
//////

var S2dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterDate('2019-12-01', '2020-03-01').filterBounds(ROI)
                  // Pre-filter to get less cloudy granules.
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',11))
                  .map(maskS2clouds);
print('Intial DS',S2dataset);

var CompletedDsS2=S2dataset.map(NDVI).map(GSAVI).map(GNDVI).map(CVI).map(NDGI)
.map(NBR).map(NDII).map(GDVI).map(MSAVI).map(DVI).map(SAVI).map(MSR).map(SI)
.select(['B2', 'B3','B4','B5','B6','B7','B8','B8A','B11','B12',
          'NDVI','GSAVI','GNDVI','CVI','NDGI','NBR','NDII','GDVII','MSAVI','DVI','SAVI','MSR'
          ,'RG','S1N','NG','S2G','NR','S2R','S1G','S2N','S1R','SWIR21']);

print('Completed Ds',CompletedDsS2);
  
var FinalDsS2=CompletedDsS2.mean();
print('Final DS',FinalDsS2);


var FinalDsS2_P1=FinalDsS2.select(['B2', 'B3','B4','B5','B6','B7','B8','B8A','B11','B12','NDVI','GNDVI',
'CVI','NDGI','NBR','NDII','GDVII','DVI','RG','S1N','NG','S2G','NR','S2R','S1G','S2N','S1R','SWIR21']);

Export.image.toDrive({
  image: FinalDsS2_P1,
  description: 'Cold_Optical_M1_P1',
  folder:'FinalMaps',
  scale: 10,
  region: ROI,
  maxPixels: 1e13,
});
var FinalDsS2_P2=FinalDsS2.select(['GSAVI','MSAVI','SAVI','MSR',]);
          
Export.image.toDrive({
  image: FinalDsS2_P2,
  description: 'Cold_Optical_M1_P2',
  folder:'FinalMaps',
  scale: 10,
  region: ROI,
  maxPixels: 1e13,
});

var CHM=ee.ImageCollection('projects/meta-forest-monitoring-okw37/assets/CanopyHeight').mosaic()
var CHM_Model=CHM.clip(ROI);

Export.image.toDrive({
  image: CHM_Model,
  description: 'CHM',
  folder:'FinalMaps',
  scale: 1,
  region: ROI,
  maxPixels: 1e13,
});