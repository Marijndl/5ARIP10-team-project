import ExtractCenterline
import re
extractLogic = ExtractCenterline.ExtractCenterlineLogic()

def resample_segment(segment, label_id, sample_number = 350):
  # Resample each segment
  currentPoints = segment.GetCurvePointsWorld()
  newPoints = vtk.vtkPoints()
  sampleDist = segment.GetCurveLengthWorld() / (sample_number - 1)
  closedCurveOption = 0
  segment.ResamplePoints(currentPoints, newPoints, sampleDist, closedCurveOption)

  vector = vtk.vtkVector3d()
  pt = [0, 0, 0]
  resampledCurve = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", f"resampledCurve{label_id}")

  for controlPoint in range(0, newPoints.GetNumberOfPoints()):
    newPoints.GetPoint(controlPoint, pt)
    vector[0] = pt[0]
    vector[1] = pt[1]
    vector[2] = pt[2]
    resampledCurve.AddControlPoint(vector)
  return resampledCurve

def split_segment(segmentationNode):
  # Initialize segment editor
  segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
  segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
  segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
  segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
  # segm_node = slicer.util.getNode(f'{label_id}.label-segmentation')
  segmentEditorWidget.setSegmentationNode(segmentationNode)
  # volume_node = slicer.util.getNode(f'{label_id}.label')
  # segmentEditorWidget.setMasterVolumeNode(volume_node)

  # Split islands to segments
  segmentEditorNode.SetSelectedSegmentID("Segment_1")
  segmentEditorWidget.setActiveEffectByName("Islands")
  effect = segmentEditorWidget.activeEffect()
  effect.setParameter("MinimumSize", "500")
  effect.setParameter("Operation", "SPLIT_ISLANDS_TO_SEGMENTS")
  segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
  segmentEditorNode.SetMaskMode(0)
  effect.self().onApply()

  # Cleanup
  slicer.mrmlScene.RemoveNode(segmentEditorNode)

def find_centerlines(segmentationNode, segmentID, vessel_id: int):

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
  endPointsMarkupsNode.RemoveAllControlPoints()
  for position in endpointPositions:
    endPointsMarkupsNode.AddControlPoint(vtk.vtkVector3d(position))

  # Extract the centerline
  centerlineCurveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", f"Centerline curve {vessel_id}")
  centerlinePolyData, voronoiDiagramPolyData = extractLogic.extractCenterline(inputSurfacePolyData,
                                                                              endPointsMarkupsNode)
  centerlinePropertiesTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "Centerline properties")
  extractLogic.createCurveTreeFromCenterline(centerlinePolyData, centerlineCurveNode, centerlinePropertiesTableNode)

def save_centerlines_to_file(label_id):
  vessel_segements = slicer.util.getNodesByClass('vtkMRMLMarkupsCurveNode')

  for i, segment in enumerate(vessel_segements):
    # Resample curve
    # resampledCurve = resample_segment(segment, label_id)

    # Save new curve to a file.
    # Pad the numbers with leading zeros
    label_padded = str(label_id).zfill(4)

    # Pad
    i_padded = str(i).zfill(2)

    slicer.modules.markups.logic().ExportControlPointsToCSV(segment, f"D:\\CTA data\\Segments original\\SegmentPoints_{label_padded}_{i_padded}.csv")

def extract_centerlines_slicer(label_id: int):
  # Clear up the scene
  slicer.mrmlScene.Clear()

  # Load the CTA data
  slicer.util.loadSegmentation(f"d:/CTA data/1-1000/{label_id}.label.nii.gz")

  # Select the segmentation node
  segmentationName = f'{label_id}.label.nii.gz'
  segmentationNode = slicer.util.getNode(segmentationName)

  # Split the segmentation in two
  split_segment(segmentationNode)

  # Find the centerlines for each vessel
  for i, segmentName in enumerate(['Segment_1', 'Segment_1_2']):
    segmentID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
    find_centerlines(segmentationNode, segmentID, i)

  # Save centerlines to file
  save_centerlines_to_file(label_id)


if __name__ == "__main__":
  for file in range(1):
    try:
      extract_centerlines_slicer(file+1)
    except:
      slicer.mrmlScene.Clear()
      pass

  print("Done")

#Instructions: Open a cmd prompt and run the following lines:
#cd C:\Users\20203226\AppData\Local\slicer.org\Slicer 5.6.1
#Slicer.exe --python-script "C:\Users\20203226\Documents\GitHub\5ARIP10-team-project\dataset creation\extract_centerline.py" --no-splash --no-main-window