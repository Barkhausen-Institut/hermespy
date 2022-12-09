[Setup]
AppName=HermesPy
AppVersion=0.3.0
AppPublisher=Barkhausen Institut gGmbH
AppPublisherURL=https://www.barkhauseninstitut.org/
AppSupportURL=https://github.com/Barkhausen-Institut/hermespy/issues
AppReadmeFile=https://hermespy.org/install.html
AppContact=jan.adler@barkhauseninstitut.org
ArchitecturesAllowed=x86 x64 arm64
WizardStyle=modern
DefaultDirName={autopf}\HermesPy
DefaultGroupName=HermesPy
UninstallDisplayIcon={app}\python.exe
OutputDir=.
LicenseFile="..\..\LICENSE"
PrivilegesRequired=lowest
ChangesEnvironment=yes

[Components]
Name: "HermesPy"; Description: "Main files of the simulator"; Types: full compact custom; Flags: fixed; ExtraDiskSpaceRequired: 524288000
Name: "Config"; Description: "Example configuration files"; Types: full

[Dirs]
Name: "{app}\config"; Components: Config

[Files]
Source: "{tmp}\embedded\*"; DestDir: "{app}"; Flags: external recursesubdirs
Source: "..\..\_examples\settings\*.yml"; DestDir: "{app}\config"; Flags: recursesubdirs; Components: Config
Source: "..\..\README.md"; DestDir: "{app}"; Flags: isreadme

[Run]
Filename: "{tmp}\redist.exe"; StatusMsg: "Installing Windows Visual Studio C++ Redistributables"; Check: VcRedistRequired; Flags: waituntilterminated
Filename: "{app}\python.exe"; Parameters: "{tmp}\get-pip.py"; StatusMsg: "Install pip"; Flags: waituntilterminated runasoriginaluser hidewizard; BeforeInstall: SetEnvPath
Filename: "{app}\Scripts\pip.exe"; Parameters: "--no-cache-dir install hermespy"; StatusMsg: "Install HermesPy from PyPI"; Flags: waituntilterminated runasoriginaluser hidewizard; Check: pypiInstall; BeforeInstall: SetEnvPath
Filename: "{app}\Scripts\pip.exe"; Parameters: "--no-cache-dir install redis"; StatusMsg: "Install missing redis"; Flags: waituntilterminated runasoriginaluser hidewizard; Check: pypiInstall; BeforeInstall: SetEnvPath
Filename: "{app}\Scripts\pip.exe"; Parameters: "--no-cache-dir install setuptools sphinx wheel pybind11 scikit-build cmake"; StatusMsg: "Install build dependencies"; Flags: shellexec waituntilterminated runasoriginaluser hidewizard; Check: repoInstall
Filename: "{app}\Scripts\pip.exe"; Parameters: "--no-cache-dir install --no-build-isolation -e {code:repoDirectory}"; StatusMsg: "Install HermesPy from local repository"; Flags: shellexec waituntilterminated runasoriginaluser hidewizard; Check: repoInstall
Filename: "{app}\Scripts\hermes.exe"; Parameters: "--help"; Description: "Run Hermes shell command"; Flags: postinstall runasoriginaluser unchecked

[Registry]
Root: HKCU; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\hermes.exe"; ValueName: "Path"; ValueType: string; ValueData: "{app}\Scripts\hermes.exe"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Environment"; ValueName: "Path"; ValueData: "{olddata}{app}\Scripts;"; ValueType: expandsz; Check: HasPath(ExpandConstant('{app}\Scripts'))

[UninstallDelete]
Type: filesandordirs; Name: "{app}\Lib"
Type: filesandordirs; Name: "{app}\Scripts"
Type: filesandordirs; Name: "{app}\share"
Type: dirifempty; Name: "{app}"

[Code]

#ifdef UNICODE
  #define AW "W"
#else
  #define AW "A"
#endif

const
  SHCONTCH_NOPROGRESSBOX = 4;
  SHCONTCH_RESPONDYESTOALL = 16;
  DEBUG = False;

var
  DownloadPage: TDownloadWizardPage;
  AdditionalInputDirPage: TInputDirWizardPage;
  InstallVcRedist: BOOL;


function SetEnvironmentVariable(lpName: string; lpValue: string): BOOL;
  external 'SetEnvironmentVariable{#AW}@kernel32.dll stdcall';

procedure SetEnvPath;
begin
  if not SetEnvironmentVariable('PATH', ExpandConstant('{app}\Scripts')) then
    MsgBox(SysErrorMessage(DLLGetLastError), mbError, MB_OK);
end;

function pypiInstall: Boolean;
begin
  Result:= not DEBUG;
end;

function repoInstall: Boolean;
begin
  Result := DEBUG;
end;

function repoDirectory(Value: string): string;
begin
  Result := AdditionalInputDirPage.Values[0];
end;

procedure UnZip(ZipPath, TargetPath: string); 
var
  Shell: Variant;
  ZipFile: Variant;
  TargetFolder: Variant;
begin
  Shell := CreateOleObject('Shell.Application');

  ZipFile := Shell.NameSpace(ZipPath);
  if VarIsClear(ZipFile) then
    RaiseException(
      Format('ZIP file "%s" does not exist or cannot be opened', [ZipPath]));

  TargetFolder := Shell.NameSpace(TargetPath);
  if VarIsClear(TargetFolder) then
    RaiseException(Format('Target path "%s" does not exist', [TargetPath]));

  TargetFolder.CopyHere(
    ZipFile.Items, SHCONTCH_NOPROGRESSBOX or SHCONTCH_RESPONDYESTOALL);
end;


function UnzipFiles() : Boolean;
begin
  CreateDir(ExpandConstant('{tmp}') + '\embedded');
  UnZip(ExpandConstant('{tmp}') + '\embedded.zip', ExpandConstant('{tmp}') + '\embedded');
  Log('Unzipped files');
end;

function PatchPath: Boolean;
var
File: String;
C: AnsiString;
CU: String;
begin
        File := ExpandConstant('{tmp}\embedded\python39._pth')
        LoadStringFromFile(File, C);
        CU := C;
        StringChangeEx(CU, '#import site', 'import site', False);
        C := CU;
        SaveStringToFile(File, C, False);          
end;


function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  if Progress = ProgressMax then
    Log(Format('Successfully downloaded file to %s: %s', [ExpandConstant('{tmp}'), FileName]));
  Result := True;
end;


function DownloadFiles() : Boolean;
begin
  DownloadPage.Clear;
  
  case ProcessorArchitecture of
    paX86: DownloadPage.Add('https://www.python.org/ftp/python/3.9.12/python-3.9.12-embed-win32.zip', 'embedded.zip', '');
    paX64: DownloadPage.Add('https://www.python.org/ftp/python/3.9.12/python-3.9.12-embed-amd64.zip', 'embedded.zip', '');
  end;   
  
  DownloadPage.Add('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py', '');

  if InstallVcRedist then begin
    case ProcessorArchitecture of
      paX86: DownloadPage.Add('https://aka.ms/vs/17/release/vc_redist.x86.exe', 'redist.exe', '');
      paX64: DownloadPage.Add('https://aka.ms/vs/17/release/vc_redist.x64.exe', 'redist.exe', '');
    end;
  end;

  DownloadPage.Show;
  try
    try
      DownloadPage.Download; // This downloads the files to {tmp}
      UnzipFiles();
      PatchPath();
      Result := True;
    except
      if DownloadPage.AbortedByUser then
        Log('Aborted by user.')
      else
        SuppressibleMsgBox(AddPeriod(GetExceptionMessage), mbCriticalError, MB_OK, IDOK);
      Result := False;
    end;
  finally
    DownloadPage.Hide;
  end;
end;

function VcRedistRequired: Boolean;
var 
  Version: String;
begin
  if RegQueryStringValue(HKEY_LOCAL_MACHINE,
       'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Version', Version) then
  begin
    // Is the installed version at least 14.14 ? 
    Log('VC Redist Version check : found ' + Version);
    Result := (CompareStr(Version, 'v14.14.26429.03')<0);
  end
  else 
  begin
    // Not even an old version installed
    Result := True;
  end;
end;

procedure InitializeWizard;
begin

  InstallVcRedist := VcRedistRequired();
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), @OnDownloadProgress);

  if DEBUG then begin
      AdditionalInputDirPage := CreateInputDirPage(wpSelectDir, 'Select HermesPy Repository Directory', 'Where is the source code located?', 'The binaries will be built from here', False, '');
      AdditionalInputDirPage.Add('');
  end;

end;


function NextButtonClick(CurPageID: Integer): Boolean;
var
  ResultCode: Integer;
begin

  if CurPageID = wpReady then begin
    
    Result := DownloadFiles();

  end else
    Result := True;
end;

function HasPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER,
    'Environment',
    'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  { look for the path with leading and trailing semicolon }
  { Pos() returns 0 if not found }
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;

procedure RemovePath(Path: string);
var
  Paths: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKCU, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Paths) then
  begin
    Log('PATH not found');
  end
    else
  begin
    Log(Format('PATH is [%s]', [Paths]));

    P := Pos(';' + Uppercase(Path) + ';', ';' + Uppercase(Paths) + ';');
    if P = 0 then
    begin
      Log(Format('Path [%s] not found in PATH', [Path]));
    end
      else
    begin
      if P > 1 then P := P - 1;
      Delete(Paths, P, Length(Path) + 1);
      Log(Format('Path [%s] removed from PATH => [%s]', [Path, Paths]));

      if RegWriteStringValue(HKCU, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Paths) then
      begin
        Log('PATH written');
      end
        else
      begin
        Log('Error writing PATH');
      end;
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
  begin
    RemovePath(ExpandConstant('{app}\Scripts'));
  end;
end;