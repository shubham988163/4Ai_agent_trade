' RanchoTrade Background Silent Launcher
' Starts trading.dashboard on port 8787 via pythonw without showing a console window.
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptPath = WScript.ScriptFullName
scriptDir = fso.GetParentFolderName(scriptPath)
projectDir = fso.GetParentFolderName(scriptDir)

WshShell.CurrentDirectory = projectDir

Dim pythonExe
If fso.FileExists(projectDir & "\.venv\Scripts\pythonw.exe") Then
    pythonExe = projectDir & "\.venv\Scripts\pythonw.exe"
Else
    pythonExe = "pythonw.exe"
End If

' Run dashboard on 8787 hidden (0 = hide window, false = do not wait)
WshShell.Run """" & pythonExe & """ -u -m trading.dashboard 8787", 0, False
