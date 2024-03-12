import ExtractCenterline
extractLogic = ExtractCenterline.ExtractCenterlineLogic()

def extract_centerlines_slicer(label_id: int):
  # Load the CTA data
  slicer.util.loadSegmentation(f"c:/Users/20203226/Documents/CTA data/1-200/{label_id}.label.nii.gz")

  # Select the segmentation node
  segmentationName = f'{label_id}.label.nii.gz' # replace with the name of your segmentation
  segmentName = 'Segment_1' # replace with the name of the segment you want to find the centerline of
  segmentationNode = slicer.util.getNode(segmentationName)
  segmentID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)

  # Preprocess the surface
  inputSurfacePolyData = extractLogic.polyDataFromNode(segmentationNode, segmentID)
  # targetNumberOfPoints = 5000.0
  # decimationAggressiveness = 4 # I had to lower this to 3.5 in at least one case to get it to work, 4 is the default in the module
  # subdivideInputSurface = False
  # preprocessedPolyData = extractLogic.preprocess(inputSurfacePolyData, targetNumberOfPoints, decimationAggressiveness, subdivideInputSurface)

  # Auto-detect the endpoints
  endPointsMarkupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Centerline endpoints")
  networkPolyData = extractLogic.extractNetwork(inputSurfacePolyData, endPointsMarkupsNode)
  startPointPosition = None
  endpointPositions = extractLogic.getEndPoints(networkPolyData, startPointPosition)
  endPointsMarkupsNode.RemoveAllMarkups()
  for position in endpointPositions:
    endPointsMarkupsNode.AddControlPoint(vtk.vtkVector3d(position))

  # Extract the centerline
  centerlineCurveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "Centerline curve")
  centerlinePolyData, voronoiDiagramPolyData = extractLogic.extractCenterline(inputSurfacePolyData, endPointsMarkupsNode)
  centerlinePropertiesTableNode = None
  extractLogic.createCurveTreeFromCenterline(centerlinePolyData, centerlineCurveNode, centerlinePropertiesTableNode)

print("Done")

if __name__ == "__main__":
  extract_centerlines_slicer(2)

#Instructions: Open a cmd prompt and run the following lines:
#cd C:\Users\20203226\AppData\Local\slicer.org\Slicer 5.6.1
#Slicer.exe --python-script "C:\Users\20203226\Documents\GitHub\5ARIP10-team-project\extract_centerline.py"