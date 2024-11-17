import os
import shutil
import subprocess
from xml.dom import minidom
import xml.etree.ElementTree as ET
import zipfile
import ifcopenshell

# config the solibri autorun workflow xml file
def set_xml_path(workflow_path, 
                    smc_path = None, 
                    old_ifc_path = None, 
                    new_ifc_path = None, 
                    bcf_export_path = None):
    tree = ET.parse(workflow_path)
    root = tree.getroot()

    openmodel = root.find(".//openmodel")
    if openmodel is not None and smc_path:
        openmodel.set("file", smc_path)
    
    updatemodel = root.find(".//updatemodel")
    if updatemodel is not None:
        if old_ifc_path:
            updatemodel.set("file", old_ifc_path)
        if new_ifc_path:
            updatemodel.set("with", new_ifc_path)
    
    bcfreport = root.find(".//bcfreport")
    if bcfreport is not None and bcf_export_path:
        bcfreport.set("file", bcf_export_path)
    
    savemodel = root.find(".//savemodel")
    if savemodel is not None and smc_path:
        savemodel.set("file", smc_path)
    
    tree.write(workflow_path, encoding="ISO-8859-1", xml_declaration=True)


def check_model(workflow_path, solibri_path=r"C:\Program Files\Solibri\SOLIBRI\Solibri.exe"):
    # Run Solibri Model Checker
    subprocess.run([solibri_path, workflow_path])

def process_bcf_report(bcf_report_path, new_ifc_path):
    # create a issue text template
    issue_template = """
- Issue: {}
  Issue description: {}
  Related element uuids: {}
    """
    # overall issue string
    issues_sum = ""
    
    # unzip the bcf report
    with zipfile.ZipFile(bcf_report_path, 'r') as zip_ref:
        folder_path = bcf_report_path.split(".")[0]
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        zip_ref.extractall(folder_path)
    # traverse all issues
    for root, dirs, files in os.walk(folder_path):
        # files share same root
        for file in files:
            # markup
            if file.endswith(".bcf"):
                bcf_file_path = os.path.join(root, file)
                markup = minidom.parse(str(bcf_file_path))
                title = markup.getElementsByTagName('Title')[0].childNodes[0].nodeValue
                description = markup.getElementsByTagName('Description')[0].childNodes[0].nodeValue if markup.getElementsByTagName('Description')[0].childNodes else '' # in case there's no description.
            # viewpoint
            if file.endswith(".bcfv"):
                uuids = []
                viewpoint_file_path = os.path.join(root, file)
                viewpoint = minidom.parse(str(viewpoint_file_path))
                components = viewpoint.getElementsByTagName('Component')
                model = ifcopenshell.open(new_ifc_path)
                for component in components:
                    guid = component.getAttribute('IfcGuid')
                    element = model.by_guid(guid)
                    uuid = element.get_info()["Description"]
                    uuids.append(uuid)
                # usually viewpoint file is after the issue file, so this works
                issues_sum += issue_template.format(title, description, uuids)
    print(issues_sum)
    return issues_sum
